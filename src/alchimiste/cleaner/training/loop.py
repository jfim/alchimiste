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
import json
import random
from dataclasses import dataclass
from pathlib import Path

import hydra
import numpy as np
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from alchimiste.cleaner.data.loader import LabeledArticle, load_oxen_tree
from alchimiste.cleaner.data.oxen_meta import OxenMeta, read_commit
from alchimiste.cleaner.data.split import SplitsManifest, make_splits
from alchimiste.cleaner.models.base import TokenTagger
from alchimiste.cleaner.training import mlflow_io
from alchimiste.cleaner.training.finalize import finalize as _finalize

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
    """Single-seed training pass. Backward-compatible wrapper around the
    multi-seed driver: if `cfg.seeds` is present and non-empty, dispatch
    to `train_seeds` (each seed gets its own sub-directory + MLflow run);
    otherwise run once at `cfg.seed`."""
    seeds = _seeds_from_cfg(cfg)
    if seeds is not None:
        results = train_seeds(cfg, seeds)
        # Return the first run's result for callers that expect a single
        # RunResult. `train_seeds` is the source of truth for the rest.
        return results[0]
    return _train_one(cfg, seed=int(cfg.seed), artifact_dir=_resolve_artifact_dir())


def train_seeds(cfg: DictConfig, seeds: list[int]) -> list[RunResult]:
    """Train one model per seed in a single Python process.

    Amortizes the per-process setup that doesn't depend on the seed:
      * Python interpreter + library imports (paid by the caller).
      * Pretrained encoder + tokenizer files in the HF cache.
      * Loading articles from the oxen working tree.
      * Tokenizing every article once.

    The encoder weights live in different Python objects per seed (a new
    `Tagger` is instantiated each time) so that an unfrozen-backbone run
    on seed `n+1` starts from the pristine pretrained state, not from
    the fine-tuned weights produced by seed `n`. Verified empirically
    with `seeds=[k,k,k]` producing identical loss curves.

    Each seed's artifacts live under `<hydra_output>/seed-<N>/`, and each
    gets its own MLflow run.
    """
    if not seeds:
        raise ValueError("train_seeds: at least one seed required")

    artifact_root = _resolve_artifact_dir()
    artifact_root.mkdir(parents=True, exist_ok=True)

    oxen_meta = _try_read_oxen_meta(Path(cfg.data.oxen_dir))
    if oxen_meta is not None and oxen_meta.dirty and not cfg.data.allow_dirty:
        raise RuntimeError(
            f"oxen working tree at {cfg.data.oxen_dir} is dirty; refuse to "
            "train (REQ-014). Re-run with data.allow_dirty=true to override."
        )
    articles = load_oxen_tree(
        cfg.data.oxen_dir,
        stage=cfg.data.stage,
        require_nfc=cfg.data.require_nfc,
        range_units=cfg.data.range_units,
        min_bytes=int(cfg.data.get("min_bytes", 0)),
    )

    # Tokenize every article exactly once. Tokenization is deterministic
    # given (text, tokenizer, max_seq_len) — no seed dependency — so the
    # cache is safe to share across seeds. Use a throwaway tagger just
    # for its tokenizer; per-seed taggers get fresh model weights below.
    _seed_all_rngs(int(seeds[0]))  # only affects tokenizer-init RNG, harmless
    tokenizer_tagger = _instantiate_tagger(cfg.model)
    all_tokenized = {
        a.item_id: tokenizer_tagger.tokenize([a], max_seq_len=cfg.data.max_seq_len)[0]
        for a in articles
    }

    results: list[RunResult] = []
    for i, seed in enumerate(seeds):
        artifact_dir = artifact_root / f"seed-{seed}-{i}"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        result = _train_one(
            cfg=cfg,
            seed=int(seed),
            artifact_dir=artifact_dir,
            oxen_meta=oxen_meta,
            articles=all_tokenized_to_articles_for_split(articles, all_tokenized),
            tokenized_by_id=all_tokenized,
        )
        results.append(result)
    return results


def all_tokenized_to_articles_for_split(
    articles: list[LabeledArticle],
    tokenized_by_id: dict,
) -> list[LabeledArticle]:
    """Keep articles in the same order, filtering to those that have a
    matching tokenized entry (in practice all do — guard for noisy data)."""
    return [a for a in articles if a.item_id in tokenized_by_id]


def _train_one(
    cfg: DictConfig,
    seed: int,
    artifact_dir: Path,
    oxen_meta: OxenMeta | None = None,
    articles: list[LabeledArticle] | None = None,
    tokenized_by_id: dict | None = None,
) -> RunResult:
    """Inner per-seed training pass. Used by both single-seed `train()`
    and the multi-seed `train_seeds()` driver. The `articles` and
    `tokenized_by_id` arguments let the multi-seed driver share work
    across seeds; left None they fall back to loading + tokenizing
    inside this call (the single-seed path)."""
    artifact_dir.mkdir(parents=True, exist_ok=True)
    # Seed every RNG that affects training BEFORE we instantiate the tagger
    # or build the data loader. See `_seed_all_rngs` for why.
    _seed_all_rngs(seed)

    if oxen_meta is None:
        oxen_meta = _try_read_oxen_meta(Path(cfg.data.oxen_dir))
        if oxen_meta is not None and oxen_meta.dirty and not cfg.data.allow_dirty:
            raise RuntimeError(
                f"oxen working tree at {cfg.data.oxen_dir} is dirty; refuse to "
                "train (REQ-014). Re-run with data.allow_dirty=true to override."
            )

    if articles is None:
        articles = load_oxen_tree(
            cfg.data.oxen_dir,
            stage=cfg.data.stage,
            require_nfc=cfg.data.require_nfc,
            range_units=cfg.data.range_units,
            min_bytes=int(cfg.data.get("min_bytes", 0)),
        )

    splits = make_splits(item_ids=[a.item_id for a in articles], seed=seed)
    splits.write_json(artifact_dir / "splits.json")
    train_arts, val_arts, test_arts = _partition_articles(articles, splits)

    # Fresh tagger every call so seed `n+1`'s training starts from the
    # pristine pretrained encoder, not from seed `n`'s fine-tuned weights.
    tagger = _instantiate_tagger(cfg.model)

    if tokenized_by_id is not None:
        # Multi-seed path: reuse pre-tokenized examples.
        train_ex = [tokenized_by_id[a.item_id] for a in train_arts]
        val_ex = [tokenized_by_id[a.item_id] for a in val_arts]
        test_ex = [tokenized_by_id[a.item_id] for a in test_arts]
    else:
        train_ex = tagger.tokenize(train_arts, max_seq_len=cfg.data.max_seq_len)
        val_ex = tagger.tokenize(val_arts, max_seq_len=cfg.data.max_seq_len)
        test_ex = tagger.tokenize(test_arts, max_seq_len=cfg.data.max_seq_len)

    # The model receives only its sub-config; expose training hyperparameters
    # via a private `_training` key so implementations can look them up
    # without parsing the whole config tree. Hydra's default struct mode
    # rejects new keys, so clone and disable it for the merged sub-config.
    model_cfg = OmegaConf.create(OmegaConf.to_container(cfg.model, resolve=True))
    model_cfg._training = OmegaConf.to_container(cfg.get("training", {}), resolve=True)

    # Route per-batch / per-epoch metrics into mlflow (REQ-015). If mlflow
    # isn't configured (e.g. unit tests setting tracking_uri="" to disable),
    # callers can call `train()` with mlflow disabled by patching this hook.
    with mlflow_io.start_run(cfg, oxen_meta=oxen_meta):
        callbacks = mlflow_io.build_callbacks()
        tagger.fit(train_ex, val_ex, model_cfg, callbacks)
        tagger.save(artifact_dir / "model")

        OmegaConf.save(cfg, artifact_dir / "config.yaml")

        metrics = _finalize(
            tagger=tagger,
            val_articles=val_arts,
            val_examples=val_ex,
            test_articles=test_arts,
            test_examples=test_ex,
            artifact_dir=artifact_dir,
            eval_cfg=cfg.eval,
        )
        scalar_metrics = {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))}
        mlflow_io.log_final(scalar_metrics, artifact_dir)

        if cfg.mlflow.get("register_model", False):
            threshold_payload = json.loads(
                (artifact_dir / "threshold.json").read_text(encoding="utf-8")
            )
            mlflow_io.log_pyfunc_model(
                artifact_dir,
                registered_model_name=cfg.mlflow.model_name,
                threshold=float(threshold_payload["threshold"]),
                threshold_iou=float(threshold_payload["iou_metric"]),
                fell_back_to_max_precision=bool(threshold_payload["fell_back_to_max_precision"]),
            )

    return RunResult(
        artifact_dir=artifact_dir,
        splits=splits,
        train_size=len(train_ex),
        val_size=len(val_ex),
        test_size=len(test_ex),
    )


def _seeds_from_cfg(cfg: DictConfig) -> list[int] | None:
    """Resolve `cfg.seeds` to a non-empty list, or None when absent/empty.

    Accepts: a list/ListConfig (`seeds=[11,17,42]`), a single int
    (`seeds=11`), or a string of comma-separated ints (`seeds=11,17,42`).
    """
    raw = cfg.get("seeds", None)
    if raw is None:
        return None
    if isinstance(raw, str):
        raw = [s for s in (s.strip() for s in raw.split(",")) if s]
    try:
        ints = [int(s) for s in raw]
    except (TypeError, ValueError) as e:
        raise ValueError(f"cfg.seeds must be int / list[int] / 'a,b,c'; got {raw!r}") from e
    return ints if ints else None


def _seed_all_rngs(seed: int) -> None:
    """Seed every RNG that affects training output.

    Covers: PyTorch global (which feeds the DataLoader's default shuffle
    sampler and dropout), NumPy (used by some HF tokenizer paths and any
    sklearn calls during eval), and Python's `random` (defensive). CUDA
    seeding piggy-backs on `torch.manual_seed` for the current device.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _try_read_oxen_meta(oxen_dir: Path) -> OxenMeta | None:
    """Best-effort `read_commit`: returns None when `oxen_dir` is not an
    oxen working tree (so unit tests against synthesized fixtures work).
    Re-raises any other failure so real misconfigurations are loud."""
    try:
        return read_commit(oxen_dir)
    except RuntimeError as e:
        if "oxen log failed" in str(e) or "oxen binary not found" in str(e):
            return None
        raise


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
    seeds = _seeds_from_cfg(cfg)
    if seeds is not None:
        results = train_seeds(cfg, seeds)
        for seed, r in zip(seeds, results, strict=True):
            print(f"seed={seed} artifact_dir: {r.artifact_dir}")
            print(f"  sizes: train={r.train_size} val={r.val_size} test={r.test_size}")
    else:
        result = train(cfg)
        print(f"artifact_dir: {result.artifact_dir}")
        print(f"sizes: train={result.train_size} val={result.val_size} test={result.test_size}")


if __name__ == "__main__":
    main()
