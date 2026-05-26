"""HuggingFace encoder + pluggable head (TASK-015, REQ-005).

This is the canonical `TokenTagger` implementation. It wraps any HF
encoder addressable by name (DistilBERT, MiniLM, ModernBERT, …) and
delegates classification to a `Head` chosen via the `model.head:` config
block — see `models/heads.py` for the available types and how to add
new ones.

Class imbalance (REQ-007 / NFR-002): a `class_weight_drop` config knob
scales the drop class in the cross-entropy loss. The default value of
5.0 is a starting point; the end-to-end run in TASK-029 will revisit.

The implementation deliberately avoids the HF `Trainer` so the per-batch
callback (REQ-015) is straightforward and so we have a single place to
tune CPU-only behavior.
"""

from __future__ import annotations

import contextlib
import json
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

from alchimiste.cleaner.data.align import (
    LABEL_DROP,
    LABEL_IGNORE,
    LABEL_KEEP,
    TokenizedExample,
    tokenize_and_align,
)
from alchimiste.cleaner.data.loader import LabeledArticle
from alchimiste.cleaner.models.base import TrainingCallbacks
from alchimiste.cleaner.models.heads import Head, build_head

if TYPE_CHECKING:
    from omegaconf import DictConfig

_NUM_LABELS = 2  # keep / drop
_METADATA_FILENAME = "alchimiste_meta.json"
_HEAD_STATE_FILENAME = "head_state.pt"


class Tagger:
    """Concrete `TokenTagger` (Protocol structural match — no inheritance)."""

    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg
        self._hf_model_name: str = cfg.hf_model_name
        trust_remote_code = bool(cfg.get("trust_remote_code", False))
        self.tokenizer = AutoTokenizer.from_pretrained(
            self._hf_model_name, use_fast=True, trust_remote_code=trust_remote_code
        )
        self.encoder = AutoModel.from_pretrained(
            self._hf_model_name, trust_remote_code=trust_remote_code
        )
        _repair_rope_buffer(self.encoder)
        self._head_cfg = _resolve_head_cfg(cfg)
        self.head: Head = build_head(
            self._head_cfg,
            hidden_size=self.encoder.config.hidden_size,
            num_labels=_NUM_LABELS,
        )

    # ------------------------------------------------------------------ #
    # Protocol surface                                                   #
    # ------------------------------------------------------------------ #

    def tokenize(
        self,
        articles: list[LabeledArticle],
        max_seq_len: int,
    ) -> list[TokenizedExample]:
        return [tokenize_and_align(a, self.tokenizer, max_seq_len=max_seq_len) for a in articles]

    def fit(
        self,
        train: list[TokenizedExample],
        val: list[TokenizedExample],
        cfg: DictConfig,
        callbacks: TrainingCallbacks,
    ) -> None:
        # `cfg` here is the model sub-config with `_training` attached by
        # the loop; the training hyperparameters live under that group.
        # Overwrite `self.cfg` with this augmented version so that
        # `_quick_val_loss` and `predict_token_probs` (called by
        # `finalize`) see the same training knobs (e.g. eval_batch_size,
        # precision) rather than falling back to their hard-coded
        # defaults via `_resolve_training_cfg`'s empty-dict fallback.
        self.cfg = cfg
        training_cfg = _resolve_training_cfg(cfg)

        device = _select_device(training_cfg.get("device", "auto"))
        self.encoder.to(device)
        self.head.to(device)

        # Optional: shrink the encoder's activation memory for long contexts.
        # Trades ~30% compute for big VRAM savings; HF's flag handles the rest.
        if bool(training_cfg.get("gradient_checkpointing", False)):
            self.encoder.gradient_checkpointing_enable()
            self.encoder.config.use_cache = False

        # Optional: freeze the encoder, train only the head.
        if bool(training_cfg.get("freeze_backbone", False)):
            for p in self.encoder.parameters():
                p.requires_grad = False

        loss_fn = _build_loss(
            class_weight_drop=float(training_cfg.get("class_weight_drop", 1.0)),
            label_smoothing=float(training_cfg.get("label_smoothing", 0.0)),
            device=device,
        )

        # `torch.compile` traces encoder+head into fused kernels. `dynamic=True`
        # tells the compiler our sequence dimension varies per batch (we pad
        # to the longest example in each batch), which avoids recompiles.
        # We compile in place; `OptimizedModule` is a transparent wrapper —
        # `.train()/.eval()/.state_dict()/.parameters()` all delegate to the
        # original module. `save()` unwraps via `_orig_mod` before calling
        # `save_pretrained`, since that HF method isn't on `OptimizedModule`.
        if bool(training_cfg.get("torch_compile", False)):
            self.encoder = torch.compile(self.encoder, dynamic=True)
            self.head = torch.compile(self.head, dynamic=True)

        batch_size = int(training_cfg.get("batch_size", 8))
        # `pin_memory` only does anything when the destination is CUDA — pinned
        # host pages let CUDA do async H→D copies. Gating on device.type avoids
        # a needless allocation overhead on CPU-only runs.
        use_pin = device.type == "cuda"
        loader = DataLoader(
            _PaddedDataset(train, pad_token_id=self.tokenizer.pad_token_id or 0),
            batch_size=batch_size,
            shuffle=True,
            collate_fn=_PaddedDataset.collate,
            pin_memory=use_pin,
        )

        # Two param groups so the freshly-init'd head can learn faster than
        # the pretrained backbone. `head_lr_multiplier=1.0` collapses to a
        # single effective LR.
        base_lr = float(training_cfg.get("learning_rate", 3.0e-5))
        head_mult = float(training_cfg.get("head_lr_multiplier", 1.0))
        weight_decay = float(training_cfg.get("weight_decay", 0.01))
        backbone_params = [p for p in self.encoder.parameters() if p.requires_grad]
        head_params = [p for p in self.head.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(
            [
                {"params": backbone_params, "lr": base_lr},
                {"params": head_params, "lr": base_lr * head_mult},
            ],
            weight_decay=weight_decay,
        )

        # Effective batch = batch_size * grad_accum_steps; the scheduler is
        # stepped per optimizer-step, not per micro-batch.
        accum_steps = max(int(training_cfg.get("grad_accum_steps", 1)), 1)
        epochs = int(training_cfg.get("epochs", 1))
        steps_per_epoch = max(len(loader) // accum_steps, 1)
        total_opt_steps = steps_per_epoch * epochs

        warmup_ratio = float(training_cfg.get("warmup_ratio", 0.0))
        scheduler = None
        if warmup_ratio > 0.0 and total_opt_steps > 0:
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=int(total_opt_steps * warmup_ratio),
                num_training_steps=total_opt_steps,
            )

        max_grad_norm = float(training_cfg.get("max_grad_norm", 0.0))

        precision = str(training_cfg.get("precision", "fp32")).lower()
        autocast_ctx, grad_scaler = _build_amp(precision, device)

        # Best-on-val checkpoint tracking. When `save_best_on_val=true`,
        # snapshot encoder + head state every time val_loss improves and
        # restore the best snapshot before this fit() returns. The default
        # (false) keeps the legacy "save last-epoch weights" behavior — fine
        # for sweep runs where we mostly look at metrics. Flip on when the
        # run is destined for deployment / inference.
        save_best_on_val = bool(training_cfg.get("save_best_on_val", False))
        best_val_loss = float("inf")
        best_encoder_state: dict | None = None
        best_head_state: dict | None = None
        best_epoch: int | None = None

        step = 0
        micro_step = 0
        optimizer.zero_grad()
        for epoch in range(epochs):
            self.encoder.train()
            self.head.train()
            epoch_loss = 0.0
            n_batches = 0
            for batch in loader:
                batch = {k: v.to(device, non_blocking=use_pin) for k, v in batch.items()}
                with autocast_ctx():
                    head_out = self._forward(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        labels=batch["labels"],
                        loss_fn=loss_fn,
                    )
                    loss = head_out.loss
                assert loss is not None  # labels were passed
                # Scale so accumulated gradients match a single forward over
                # the effective batch.
                scaled_loss = loss / accum_steps
                if grad_scaler is not None:
                    grad_scaler.scale(scaled_loss).backward()
                else:
                    scaled_loss.backward()

                micro_step += 1
                if micro_step % accum_steps == 0:
                    if max_grad_norm > 0.0:
                        if grad_scaler is not None:
                            grad_scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            list(self.encoder.parameters()) + list(self.head.parameters()),
                            max_grad_norm,
                        )
                    if grad_scaler is not None:
                        grad_scaler.step(optimizer)
                        grad_scaler.update()
                    else:
                        optimizer.step()
                    if scheduler is not None:
                        scheduler.step()
                    optimizer.zero_grad()

                # `.item()` forces a GPU→CPU sync; cache the value so the
                # logging callback and the epoch accumulator share one sync
                # per step instead of two.
                loss_val = float(loss.item())
                callbacks.on_batch_end(step, loss_val)
                epoch_loss += loss_val
                n_batches += 1
                step += 1

            mean_epoch_loss = epoch_loss / max(n_batches, 1)
            val_metrics = self._quick_val_loss(val, loss_fn, device)
            # Keys are namespaced explicitly. The mlflow callback applies
            # the default `val/` namespace to bare keys, so a slash here
            # keeps the per-epoch series in their correct ns (design.md
            # § 5.4: train/epoch_loss vs val/val_loss).
            callbacks.on_epoch_end(
                epoch,
                {"train/epoch_loss": mean_epoch_loss, **val_metrics},
            )

            if save_best_on_val:
                v = float(val_metrics.get("val_loss", float("nan")))
                # NaN never compares strictly less than anything, so a bad
                # epoch (e.g. empty val set) skips the checkpoint cleanly.
                if v < best_val_loss:
                    best_val_loss = v
                    # CPU clone so the snapshot doesn't fight live training
                    # tensors for GPU memory across the rest of the run.
                    best_encoder_state = {
                        k: t.detach().cpu().clone() for k, t in self.encoder.state_dict().items()
                    }
                    best_head_state = {
                        k: t.detach().cpu().clone() for k, t in self.head.state_dict().items()
                    }
                    best_epoch = epoch

        if save_best_on_val and best_encoder_state is not None and best_head_state is not None:
            # Restore the best-on-val weights so subsequent predict / save
            # calls reflect them, not the final-epoch state.
            self.encoder.load_state_dict(best_encoder_state)
            self.head.load_state_dict(best_head_state)
            callbacks.on_epoch_end(
                # Re-emit at a clearly distinguishable step so the time
                # series shows where the restored checkpoint came from.
                epochs,
                {"val/best_val_loss": best_val_loss, "val/best_epoch": float(best_epoch or 0)},
            )

    def predict_token_probs(
        self,
        examples: list[TokenizedExample],
    ) -> list[list[float]]:
        if not examples:
            return []
        device = next(self.encoder.parameters()).device
        self.encoder.eval()
        self.head.eval()
        # Reuse the training collate so we get padded batches. Order is
        # preserved (shuffle=False); per-example lengths are recovered from
        # the original `examples` list to strip padding from the outputs.
        # An isolated generator keeps DataLoader's base-seed draw from
        # nudging the global RNG (which would otherwise perturb the
        # training-time shuffle on subsequent epochs).
        batch_size = int(_resolve_training_cfg(self.cfg).get("eval_batch_size", 32))
        loader = DataLoader(
            _PaddedDataset(examples, pad_token_id=self.tokenizer.pad_token_id or 0),
            batch_size=batch_size,
            shuffle=False,
            collate_fn=_PaddedDataset.collate,
            generator=torch.Generator(),
        )
        # Match the training-time precision so encoder activations
        # don't bloat to fp32 in eval. At long contexts (e.g. ModernBERT
        # at sl=8192) the fp32 attention scores OOM on a 12 GB card even
        # at bs=1; bf16 halves the peak.
        precision = str(_resolve_training_cfg(self.cfg).get("precision", "fp32")).lower()
        autocast_ctx, _ = _build_amp(precision, device)
        out: list[list[float]] = []
        idx = 0
        with torch.no_grad(), autocast_ctx():
            for batch in loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                head_out = self._forward(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                )
                # CRF heads emit raw transition-scoring logits; the per-token
                # drop probability is the CRF *marginal* P(y_t = drop | x), not
                # softmax-of-emissions. For non-CRF heads the two are
                # equivalent. `getattr` keeps this safe for the linear and
                # plain-conv heads that don't expose a CRF attribute.
                crf = getattr(self.head, "crf", None)
                if crf is not None:
                    probs = crf.marginals(
                        head_out.logits.float(), batch["attention_mask"]
                    )[..., LABEL_DROP]
                else:
                    probs = torch.softmax(head_out.logits, dim=-1)[..., LABEL_DROP]
                probs_list = probs.tolist()
                for row in probs_list:
                    n = len(examples[idx].input_ids)
                    out.append([float(p) for p in row[:n]])
                    idx += 1
        return out

    def save(self, dst: Path) -> None:
        dst.mkdir(parents=True, exist_ok=True)
        # If the encoder/head were wrapped by `torch.compile`, peel back to
        # the underlying HF model so `save_pretrained` is available and the
        # checkpoint is portable across runs that don't compile.
        encoder = getattr(self.encoder, "_orig_mod", self.encoder)
        head = getattr(self.head, "_orig_mod", self.head)
        encoder.save_pretrained(dst)
        self.tokenizer.save_pretrained(dst)
        torch.save(head.state_dict(), dst / _HEAD_STATE_FILENAME)
        (dst / _METADATA_FILENAME).write_text(
            json.dumps(
                {
                    "hf_model_name": self._hf_model_name,
                    "head": dict(self._head_cfg),
                    "trust_remote_code": bool(self.cfg.get("trust_remote_code", False)),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls, src: Path) -> Tagger:
        meta = json.loads((src / _METADATA_FILENAME).read_text(encoding="utf-8"))
        from omegaconf import DictConfig as _DC  # local to avoid heavy import at module load

        # Reconstruct the model sub-config and instantiate via __init__, then
        # swap weights in. `head` defaults to `{type: linear}` so artifacts
        # produced by the pre-refactor code (no head block in meta) load
        # with the implicit linear head their AutoModelForTokenClassification
        # checkpoint encoded.
        trust_remote_code = bool(meta.get("trust_remote_code", False))
        cfg = _DC({
            "hf_model_name": meta["hf_model_name"],
            "head": meta.get("head", {"type": "linear"}),
            "trust_remote_code": trust_remote_code,
        })
        instance = cls(cfg)
        instance.tokenizer = AutoTokenizer.from_pretrained(
            src, use_fast=True, trust_remote_code=trust_remote_code
        )
        instance.encoder = AutoModel.from_pretrained(src, trust_remote_code=trust_remote_code)
        _repair_rope_buffer(instance.encoder)
        head_state_path = src / _HEAD_STATE_FILENAME
        if head_state_path.exists():
            instance.head.load_state_dict(
                torch.load(head_state_path, map_location="cpu", weights_only=True)
            )
        return instance

    # ------------------------------------------------------------------ #
    # Helpers                                                            #
    # ------------------------------------------------------------------ #

    def _forward(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None = None,
        loss_fn: nn.Module | None = None,
    ):
        # When the backbone is frozen, autograd would still save every
        # layer's activations because the encoder output participates in
        # the head's backward graph. At 28 layers × long context that's
        # the difference between fitting and OOMing on a 12 GB card.
        # `no_grad` + `detach` lets us run the encoder as a pure feature
        # extractor; the head's own params still get gradients normally.
        encoder_frozen = not any(p.requires_grad for p in self.encoder.parameters())
        if encoder_frozen:
            with torch.no_grad():
                hidden = self.encoder(
                    input_ids=input_ids, attention_mask=attention_mask
                ).last_hidden_state
            hidden = hidden.detach()
        else:
            hidden = self.encoder(
                input_ids=input_ids, attention_mask=attention_mask
            ).last_hidden_state
        return self.head(hidden, attention_mask, labels=labels, loss_fn=loss_fn)

    def _quick_val_loss(
        self,
        val: list[TokenizedExample],
        loss_fn: nn.Module,
        device: torch.device,
    ) -> dict[str, float]:
        if not val:
            return {"val_loss": float("nan")}
        self.encoder.eval()
        self.head.eval()
        # Preserve the pre-batching metric: mean of per-example mean-over-
        # tokens losses (when val ran with batch_size=1 the batch loss WAS
        # the per-example loss, and we averaged those). Computing CE with
        # reduction='none' lets us reduce per-example, then average across
        # examples, exactly matching the old behavior.
        batch_size = int(_resolve_training_cfg(self.cfg).get("eval_batch_size", 32))
        loader = DataLoader(
            _PaddedDataset(val, pad_token_id=self.tokenizer.pad_token_id or 0),
            batch_size=batch_size,
            shuffle=False,
            collate_fn=_PaddedDataset.collate,
            # Isolated generator: keeps the eval DataLoader's base-seed draw
            # from advancing the global RNG mid-fit, which would otherwise
            # perturb the next epoch's training shuffle.
            generator=torch.Generator(),
        )
        per_example_loss_fn = nn.CrossEntropyLoss(
            weight=loss_fn.weight,
            ignore_index=LABEL_IGNORE,
            label_smoothing=loss_fn.label_smoothing,
            reduction="none",
        )
        # Match the training-time precision so encoder activations don't
        # bloat to fp32 in eval (same rationale as `predict_token_probs`).
        precision = str(_resolve_training_cfg(self.cfg).get("precision", "fp32")).lower()
        autocast_ctx, _ = _build_amp(precision, device)
        total = 0.0
        n = 0
        with torch.no_grad(), autocast_ctx():
            for batch in loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                head_out = self._forward(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                )
                # logits: (B, T, C); labels: (B, T)
                logits = head_out.logits
                labels = batch["labels"]
                per_tok = per_example_loss_fn(
                    logits.reshape(-1, logits.size(-1)),
                    labels.reshape(-1),
                ).view(labels.shape)
                valid = (labels != LABEL_IGNORE).to(per_tok.dtype)
                per_ex_sum = (per_tok * valid).sum(dim=1)
                per_ex_count = valid.sum(dim=1).clamp_min(1.0)
                per_ex_mean = per_ex_sum / per_ex_count
                total += float(per_ex_mean.sum().item())
                n += per_ex_mean.numel()
        return {"val_loss": total / max(n, 1)}


# ---------------------------------------------------------------------------- #
# Module-private helpers                                                       #
# ---------------------------------------------------------------------------- #


def _resolve_training_cfg(model_cfg: DictConfig) -> DictConfig | dict:
    """The model is handed only its sub-config; for training hyperparameters
    we look at a sibling `training` group attached by the loop (if any) or
    fall back to baked-in defaults. The loop attaches it as `_training`."""
    return getattr(model_cfg, "_training", {}) or {}


def _resolve_head_cfg(model_cfg: DictConfig) -> dict:
    """Plain dict so we can json-serialize it into the saved metadata."""
    from omegaconf import OmegaConf

    head = getattr(model_cfg, "head", None)
    if head is None:
        return {"type": "linear"}
    if hasattr(head, "_content") or hasattr(head, "keys"):
        try:
            return dict(OmegaConf.to_container(head, resolve=True))  # type: ignore[arg-type]
        except Exception:
            return dict(head)
    return dict(head)


def _repair_rope_buffer(encoder: nn.Module) -> None:
    # Workaround for NeoBERT (chandar-lab/NeoBERT): its modeling code
    # registers `freqs_cis` as a non-persistent buffer in __init__, but the
    # current transformers fast-init path leaves that buffer's storage
    # uninitialised after `from_pretrained` — every forward then produces
    # NaN. We recompute the buffer from the model's own rotary module.
    # No-op for any encoder that doesn't expose a `freqs_cis` attribute.
    fc = getattr(encoder, "freqs_cis", None)
    if not isinstance(fc, torch.Tensor):
        return
    try:
        import importlib

        rotary = importlib.import_module(encoder.__class__.__module__.rsplit(".", 1)[0] + ".rotary")
        precompute = rotary.precompute_freqs_cis
    except (ImportError, AttributeError):
        return
    cfg = encoder.config
    fresh = precompute(cfg.hidden_size // cfg.num_attention_heads, cfg.max_length)
    encoder.freqs_cis = fresh.to(device=fc.device, dtype=fc.dtype)


def _select_device(device_str: str) -> torch.device:
    if device_str == "cpu":
        return torch.device("cpu")
    if device_str == "cuda":
        return torch.device("cuda")
    # "auto"
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _build_loss(
    *,
    class_weight_drop: float,
    label_smoothing: float,
    device: torch.device,
) -> nn.Module:
    """Per-token weighted cross-entropy. The drop class gets a configurable
    multiplier (REQ-007); the keep class is weight 1.0; LABEL_IGNORE tokens
    are skipped via the standard `ignore_index`. Optional label smoothing
    softens hard 0/1 targets — useful when boundary tokens are noisily
    labeled."""
    weights = torch.tensor([1.0, float(class_weight_drop)], device=device)
    return nn.CrossEntropyLoss(
        weight=weights,
        ignore_index=LABEL_IGNORE,
        label_smoothing=float(label_smoothing),
    )


def _build_amp(precision: str, device: torch.device):
    """Return (autocast_context_factory, grad_scaler_or_None) for the
    chosen mixed-precision mode. fp32 is a no-op; bf16 needs no scaler;
    fp16 requires a GradScaler to avoid gradient underflow."""
    if precision == "bf16":
        ctx = lambda: torch.autocast(device_type=device.type, dtype=torch.bfloat16)  # noqa: E731
        return ctx, None
    if precision == "fp16":
        ctx = lambda: torch.autocast(device_type=device.type, dtype=torch.float16)  # noqa: E731
        scaler = torch.amp.GradScaler(device.type) if device.type == "cuda" else None
        return ctx, scaler
    return contextlib.nullcontext, None


class _PaddedDataset(Dataset):
    """Minimal padding-collate dataset; HF Trainer's DataCollator would work
    but pulls in more surface than we need for a few hundred articles."""

    def __init__(self, examples: list[TokenizedExample], *, pad_token_id: int) -> None:
        self.examples = examples
        self.pad_token_id = pad_token_id

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> TokenizedExample:
        return self.examples[idx]

    @staticmethod
    def collate(batch: list[TokenizedExample]) -> dict[str, torch.Tensor]:
        max_len = max(len(e.input_ids) for e in batch)
        input_ids = []
        attention = []
        labels = []
        for e in batch:
            pad_n = max_len - len(e.input_ids)
            input_ids.append(list(e.input_ids) + [0] * pad_n)
            attention.append([1] * len(e.input_ids) + [0] * pad_n)
            labels.append(list(e.labels) + [LABEL_IGNORE] * pad_n)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


# Hint for the training loop's `getattr(module, "MODEL_CLASS", None)` lookup.
MODEL_CLASS = Tagger


# Re-export label constants so callers don't need a second import.
__all__ = ["LABEL_DROP", "LABEL_IGNORE", "LABEL_KEEP", "MODEL_CLASS", "Tagger"]
