"""HuggingFace encoder + 2-class token-classification head (TASK-015, REQ-005).

This is the first concrete `TokenTagger` implementation. It wraps any
HF encoder addressable by name (DistilBERT, MiniLM, ModernBERT, …) with
a per-token keep/drop head and trains it via plain PyTorch.

Class imbalance (REQ-007 / NFR-002): a `class_weight_drop` config knob
scales the drop class in the cross-entropy loss. The default value of
5.0 is a starting point; the end-to-end run in TASK-029 will revisit.

The implementation deliberately avoids the HF `Trainer` so the per-batch
callback (REQ-015) is straightforward and so we have a single place to
tune CPU-only behavior.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForTokenClassification, AutoTokenizer

from alchimiste.cleaner.data.align import (
    LABEL_DROP,
    LABEL_IGNORE,
    LABEL_KEEP,
    TokenizedExample,
    tokenize_and_align,
)
from alchimiste.cleaner.data.loader import LabeledArticle
from alchimiste.cleaner.models.base import TrainingCallbacks

if TYPE_CHECKING:
    from omegaconf import DictConfig

_NUM_LABELS = 2  # keep / drop
_METADATA_FILENAME = "alchimiste_meta.json"


class Tagger:
    """Concrete `TokenTagger` (Protocol structural match — no inheritance)."""

    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg
        self._hf_model_name: str = cfg.hf_model_name
        self.tokenizer = AutoTokenizer.from_pretrained(self._hf_model_name, use_fast=True)
        self.model = AutoModelForTokenClassification.from_pretrained(
            self._hf_model_name,
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
        self.model.to(device)

        loss_fn = _build_loss(
            class_weight_drop=float(training_cfg.get("class_weight_drop", 1.0)),
            device=device,
        )

        loader = DataLoader(
            _PaddedDataset(train, pad_token_id=self.tokenizer.pad_token_id or 0),
            batch_size=int(training_cfg.get("batch_size", 8)),
            shuffle=True,
            collate_fn=_PaddedDataset.collate,
        )

        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=float(training_cfg.get("learning_rate", 3.0e-5)),
        )

        epochs = int(training_cfg.get("epochs", 1))
        step = 0
        for epoch in range(epochs):
            self.model.train()
            epoch_loss = 0.0
            n_batches = 0
            for batch in loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                logits = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                ).logits  # (B, T, 2)
                loss = loss_fn(logits.view(-1, _NUM_LABELS), batch["labels"].view(-1))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                callbacks.on_batch_end(step, float(loss.item()))
                epoch_loss += float(loss.item())
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

    def predict_token_probs(
        self,
        examples: list[TokenizedExample],
    ) -> list[list[float]]:
        if not examples:
            return []
        device = next(self.model.parameters()).device
        self.model.eval()
        out: list[list[float]] = []
        with torch.no_grad():
            for ex in examples:
                input_ids = torch.tensor([ex.input_ids], device=device)
                attn = torch.ones_like(input_ids)
                logits = self.model(input_ids=input_ids, attention_mask=attn).logits
                probs = torch.softmax(logits, dim=-1)[0, :, LABEL_DROP]
                out.append([float(p) for p in probs.tolist()])
        return out

    def save(self, dst: Path) -> None:
        dst.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(dst)
        self.tokenizer.save_pretrained(dst)
        (dst / _METADATA_FILENAME).write_text(
            json.dumps({"hf_model_name": self._hf_model_name}, indent=2) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls, src: Path) -> Tagger:
        meta = json.loads((src / _METADATA_FILENAME).read_text(encoding="utf-8"))
        # Build via __init__ to get a fully-formed Tagger, then swap the
        # weights in from `src`.
        from omegaconf import DictConfig as _DC  # local to avoid heavy import at module load

        instance = cls(_DC({"hf_model_name": meta["hf_model_name"]}))
        instance.tokenizer = AutoTokenizer.from_pretrained(src, use_fast=True)
        instance.model = AutoModelForTokenClassification.from_pretrained(src)
        return instance

    # ------------------------------------------------------------------ #
    # Helpers                                                            #
    # ------------------------------------------------------------------ #

    def _quick_val_loss(
        self,
        val: list[TokenizedExample],
        loss_fn: nn.Module,
        device: torch.device,
    ) -> dict[str, float]:
        if not val:
            return {"val_loss": float("nan")}
        self.model.eval()
        total = 0.0
        n = 0
        with torch.no_grad():
            for ex in val:
                input_ids = torch.tensor([ex.input_ids], device=device)
                labels = torch.tensor([ex.labels], device=device)
                attn = torch.ones_like(input_ids)
                logits = self.model(input_ids=input_ids, attention_mask=attn).logits
                loss = loss_fn(logits.view(-1, _NUM_LABELS), labels.view(-1))
                total += float(loss.item())
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


def _select_device(device_str: str) -> torch.device:
    if device_str == "cpu":
        return torch.device("cpu")
    if device_str == "cuda":
        return torch.device("cuda")
    # "auto"
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _build_loss(*, class_weight_drop: float, device: torch.device) -> nn.Module:
    """Per-token weighted cross-entropy. The drop class gets a configurable
    multiplier (REQ-007); the keep class is weight 1.0; LABEL_IGNORE tokens
    are skipped via the standard `ignore_index`."""
    weights = torch.tensor([1.0, float(class_weight_drop)], device=device)
    return nn.CrossEntropyLoss(weight=weights, ignore_index=LABEL_IGNORE)


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
