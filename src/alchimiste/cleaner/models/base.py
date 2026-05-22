"""Architecture-agnostic interface for token-tagging cleaner models (REQ-006).

Every concrete model under `alchimiste.cleaner.models.*` implements
`TokenTagger`. The training loop (TASK-013) dynamically imports the
module named in `cfg.model.module` and instantiates its `TokenTagger`,
so adding a new architecture is one new module + one new yaml under
`configs/model/` — no edits anywhere else.

The protocol returns per-token *probabilities*, not hard labels, so the
threshold-selection logic (design.md § 5.3 / TASK-020) lives in one
place rather than inside each model implementation.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from omegaconf import DictConfig

    from alchimiste.cleaner.data.align import TokenizedExample
    from alchimiste.cleaner.data.loader import LabeledArticle


@dataclass
class TrainingCallbacks:
    """Hooks the model implementation MUST call so live mlflow logging works
    regardless of framework (REQ-015).

    Both hooks default to no-ops so a test can wire one stub and ignore the
    other. The training loop installs real implementations that route to
    `training/mlflow_io.py` (TASK-018).
    """

    on_batch_end: Callable[[int, float], None] = field(default=lambda step, loss: None)
    on_epoch_end: Callable[[int, dict[str, float]], None] = field(
        default=lambda epoch, metrics: None
    )


@runtime_checkable
class TokenTagger(Protocol):
    """Common interface for every concrete cleaner architecture (REQ-006).

    Implementations are free to use any framework (PyTorch, JAX, sklearn)
    — only this surface is fixed.
    """

    def tokenize(
        self,
        articles: list[LabeledArticle],
        max_seq_len: int,
    ) -> list[TokenizedExample]:
        """Tokenize `articles` using the implementation's own tokenizer.

        The training loop calls this once per partition so the loop never
        needs to know anything about the tokenizer family (HF fast,
        sentencepiece, custom). Implementations typically delegate to
        `alchimiste.cleaner.data.align.tokenize_and_align`.
        """

    def fit(
        self,
        train: list[TokenizedExample],
        val: list[TokenizedExample],
        cfg: DictConfig,
        callbacks: TrainingCallbacks,
    ) -> None:
        """Train on `train`, monitor on `val`. `cfg` is the Hydra sub-tree for
        this architecture (i.e. `cfg.model.*`). `callbacks` MUST be invoked
        per batch and per epoch so the training loop can stream metrics to
        mlflow (REQ-015)."""

    def predict_token_probs(self, examples: list[TokenizedExample]) -> list[list[float]]:
        """Return per-token drop-class probabilities (one list per example,
        same length as `example.input_ids`)."""

    def save(self, dst: Path) -> None:
        """Persist weights + tokenizer config to `dst` (created if missing)."""

    @classmethod
    def load(cls, src: Path) -> TokenTagger:
        """Inverse of `save` — reload an instance from a saved directory."""
