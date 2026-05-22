"""Pluggable classification heads on top of an HF encoder's hidden states.

The encoder modules produce `(B, T, hidden_size)` token reps; a head
turns those into per-token logits (and, in the CRF case, also owns the
loss + decode). Splitting the head from the encoder lets us compare
linear vs CRF vs Conv1D heads with a single config switch — see the
`models/encoder_hf.py` Tagger for the integration point.

Adding a new head:
  1. Subclass `Head`, implement `forward(hidden, mask, labels, loss_fn)`.
  2. Register it in `build_head`.
  3. Anything head-specific that needs persisting (e.g. CRF transition
     matrix) is just part of the head's `state_dict`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch import nn

if TYPE_CHECKING:
    from omegaconf import DictConfig


@dataclass
class HeadOutput:
    """What a head returns. `loss` is populated only when `labels` and a
    `loss_fn` are passed; otherwise the caller is just doing inference."""

    logits: torch.Tensor  # (B, T, num_labels)
    loss: torch.Tensor | None = None


class Head(nn.Module):
    """Base class — subclasses own forward + (optionally) loss/decode.

    The unit of pluggability is the *head as a whole*, not a stack of
    layers: a CRF replaces both the per-token loss and the decode step,
    so a "layer stack" abstraction would leak. Each Head subclass is
    self-contained.
    """

    num_labels: int

    def forward(  # type: ignore[override]
        self,
        hidden: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None = None,
        *,
        loss_fn: nn.Module | None = None,
    ) -> HeadOutput:
        raise NotImplementedError


class LinearHead(Head):
    """Dropout + linear projection — the default. Mirrors the head HF's
    `XxxForTokenClassification` puts on its encoders, so swapping
    backbones (distilbert -> ModernBERT etc.) is a no-op."""

    def __init__(self, *, hidden_size: int, num_labels: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.num_labels = num_labels
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(  # type: ignore[override]
        self,
        hidden: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None = None,
        *,
        loss_fn: nn.Module | None = None,
    ) -> HeadOutput:
        logits = self.classifier(self.dropout(hidden))
        loss = None
        if labels is not None and loss_fn is not None:
            loss = loss_fn(logits.view(-1, self.num_labels), labels.view(-1))
        return HeadOutput(logits=logits, loss=loss)


def build_head(cfg: DictConfig | dict, *, hidden_size: int, num_labels: int) -> Head:
    """Instantiate a head from a `head:` config block. New head types
    register here; the config schema is whatever the head's `__init__`
    accepts, accessed via `cfg.get("…")`."""
    head_type = str((cfg.get("type") if cfg else None) or "linear").lower()
    if head_type == "linear":
        return LinearHead(
            hidden_size=hidden_size,
            num_labels=num_labels,
            dropout=float(cfg.get("dropout", 0.1)) if cfg else 0.1,
        )
    raise ValueError(f"unknown head type: {head_type!r}")


__all__ = ["Head", "HeadOutput", "LinearHead", "build_head"]
