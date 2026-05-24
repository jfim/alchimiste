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
        self.tokenizer = AutoTokenizer.from_pretrained(self._hf_model_name, use_fast=True)
        self.encoder = AutoModel.from_pretrained(self._hf_model_name)
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
        # `cfg` here is the model sub-config; the training hyperparameters
        # live under the top-level `training` group. Look them up via the
        # config's parent if available; otherwise fall back to defaults.
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

        batch_size = int(training_cfg.get("batch_size", 8))
        loader = DataLoader(
            _PaddedDataset(train, pad_token_id=self.tokenizer.pad_token_id or 0),
            batch_size=batch_size,
            shuffle=True,
            collate_fn=_PaddedDataset.collate,
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
                batch = {k: v.to(device) for k, v in batch.items()}
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
        out: list[list[float]] = []
        with torch.no_grad():
            for ex in examples:
                input_ids = torch.tensor([ex.input_ids], device=device)
                attn = torch.ones_like(input_ids)
                head_out = self._forward(input_ids=input_ids, attention_mask=attn)
                probs = torch.softmax(head_out.logits, dim=-1)[0, :, LABEL_DROP]
                out.append([float(p) for p in probs.tolist()])
        return out

    def save(self, dst: Path) -> None:
        dst.mkdir(parents=True, exist_ok=True)
        self.encoder.save_pretrained(dst)
        self.tokenizer.save_pretrained(dst)
        torch.save(self.head.state_dict(), dst / _HEAD_STATE_FILENAME)
        (dst / _METADATA_FILENAME).write_text(
            json.dumps(
                {"hf_model_name": self._hf_model_name, "head": dict(self._head_cfg)},
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
        cfg = _DC({
            "hf_model_name": meta["hf_model_name"],
            "head": meta.get("head", {"type": "linear"}),
        })
        instance = cls(cfg)
        instance.tokenizer = AutoTokenizer.from_pretrained(src, use_fast=True)
        instance.encoder = AutoModel.from_pretrained(src)
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
        enc = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        return self.head(enc.last_hidden_state, attention_mask, labels=labels, loss_fn=loss_fn)

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
        total = 0.0
        n = 0
        with torch.no_grad():
            for ex in val:
                input_ids = torch.tensor([ex.input_ids], device=device)
                labels = torch.tensor([ex.labels], device=device)
                attn = torch.ones_like(input_ids)
                head_out = self._forward(
                    input_ids=input_ids,
                    attention_mask=attn,
                    labels=labels,
                    loss_fn=loss_fn,
                )
                assert head_out.loss is not None
                total += float(head_out.loss.item())
                n += 1
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
