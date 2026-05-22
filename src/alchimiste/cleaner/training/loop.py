"""Architecture-agnostic training loop (TASK-013, REQ-005, REQ-006).

The loop is deliberately thin: it loads data, splits it, hands tokenized
examples to whatever `TokenTagger` implementation `cfg.model.module`
points at, and writes per-run artifacts under the Hydra runtime
`output_dir`. mlflow logging is wired in TASK-018; threshold selection
in TASK-020.

NFR-008 (parallel experiments): every output path derives from
`HydraConfig.get().runtime.output_dir`, which Hydra makes unique per run
(typically `outputs/<timestamp>/`). No code path here writes to a shared
mutable location.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from alchimiste.cleaner.data.loader import LabeledArticle, load_oxen_tree
from alchimiste.cleaner.data.split import SplitsManifest, make_splits
from alchimiste.cleaner.models.base import TokenTagger, TrainingCallbacks

_CONFIG_PATH = str(Path(__file__).resolve().parents[4] / "configs")


@dataclass(frozen=True)
class RunResult:
    """What the training loop returns to its caller.

    The loop's only obligation is to leave a self-contained `artifact_dir`
    on disk; everything else is observable from there.
    """

    artifact_dir: Path
    splits: SplitsManifest
    train_size: int
    val_size: int
    test_size: int


def train(cfg: DictConfig) -> RunResult:
    """Run one training pass. Pure-function-ish: I/O is confined to
    `artifact_dir` (derived from Hydra) and the model's `fit` callback
    chain."""
    artifact_dir = _resolve_artifact_dir()
    artifact_dir.mkdir(parents=True, exist_ok=True)

    tagger = _instantiate_tagger(cfg.model)
    articles = load_oxen_tree(
        cfg.data.oxen_dir,
        stage=cfg.data.stage,
        require_nfc=cfg.data.require_nfc,
        range_units=cfg.data.range_units,
    )

    splits = make_splits(
        item_ids=[a.item_id for a in articles],
        seed=cfg.seed,
    )
    splits.write_json(artifact_dir / "splits.json")

    train_arts, val_arts, test_arts = _partition_articles(articles, splits)

    train_ex = tagger.tokenize(train_arts, max_seq_len=cfg.data.max_seq_len)
    val_ex = tagger.tokenize(val_arts, max_seq_len=cfg.data.max_seq_len)
    test_ex = tagger.tokenize(test_arts, max_seq_len=cfg.data.max_seq_len)

    # TASK-018 will replace these with real mlflow-routing callbacks.
    callbacks = TrainingCallbacks()

    # The model receives only its sub-config; expose training hyperparameters
    # via a private `_training` key so implementations can look them up
    # without parsing the whole config tree.
    model_cfg = OmegaConf.merge(cfg.model, {"_training": cfg.get("training", {})})
    tagger.fit(train_ex, val_ex, model_cfg, callbacks)

    # Smoke-call predict on val so a broken predict_token_probs surfaces
    # immediately, not at eval time.
    _ = tagger.predict_token_probs(val_ex)

    tagger.save(artifact_dir / "model")

    # Persist the resolved config alongside the artifact so it can be
    # re-loaded without re-composing (REQ-011 artifact layout).
    OmegaConf.save(cfg, artifact_dir / "config.yaml")

    return RunResult(
        artifact_dir=artifact_dir,
        splits=splits,
        train_size=len(train_ex),
        val_size=len(val_ex),
        test_size=len(test_ex),
    )


def _instantiate_tagger(model_cfg: DictConfig) -> TokenTagger:
    """Dynamically import `cfg.model.module` and instantiate its `TokenTagger`.

    Convention: the module exposes a class named `Tagger` (the canonical
    entry point) and the class's `__init__` takes the model sub-config.
    Falls back to `MODEL_CLASS` attribute lookup so future architectures
    can override the class name.
    """
    module = importlib.import_module(model_cfg.module)
    cls = getattr(module, "MODEL_CLASS", None) or module.Tagger
    return cls(model_cfg)


def _partition_articles(
    articles: list[LabeledArticle],
    splits: SplitsManifest,
) -> tuple[list[LabeledArticle], list[LabeledArticle], list[LabeledArticle]]:
    train_ids = set(splits.train)
    val_ids = set(splits.val)
    train_arts = [a for a in articles if a.item_id in train_ids]
    val_arts = [a for a in articles if a.item_id in val_ids]
    test_arts = [a for a in articles if a.item_id in set(splits.test)]
    return train_arts, val_arts, test_arts


def _resolve_artifact_dir() -> Path:
    """Hydra runtime output dir — unique per run (NFR-008)."""
    return Path(HydraConfig.get().runtime.output_dir)


@hydra.main(config_path=_CONFIG_PATH, config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    result = train(cfg)
    # Plain stdout so `just train` is grep-able / scriptable. Real
    # observability lives in mlflow once TASK-018 lands.
    print(f"artifact_dir: {result.artifact_dir}")
    print(f"sizes: train={result.train_size} val={result.val_size} test={result.test_size}")


if __name__ == "__main__":
    main()
