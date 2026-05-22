"""mlflow integration helpers for the training loop (TASK-018, REQ-015, NFR-007).

Centralises every interaction with mlflow so the training loop and the
end-of-run finalize code share one source of truth for metric names and
tag keys (see design.md § 5.4).
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import mlflow

from alchimiste.cleaner.data.oxen_meta import OxenMeta
from alchimiste.cleaner.models.base import TrainingCallbacks

if TYPE_CHECKING:
    from omegaconf import DictConfig


# Run tag keys. Bundled here so callers don't string-literal them at the
# call site (and so search/replace covers every reference if we rename).
TAG_DATASET_OXEN_COMMIT = "alchimiste.dataset.oxen_commit"
TAG_DATASET_DIRTY = "alchimiste.dataset.dirty"
TAG_DATASET_OXEN_DIR = "alchimiste.dataset.oxen_dir"
TAG_MODEL_ARCHITECTURE = "alchimiste.model.architecture"
TAG_THRESHOLD_VALUE = "alchimiste.threshold.value"
TAG_THRESHOLD_IOU = "alchimiste.threshold.iou"
TAG_THRESHOLD_FALLBACK = "alchimiste.threshold.fell_back_to_max_precision"


@dataclass
class RunHandle:
    """What `start_run` hands back: an active mlflow run id plus a handle
    to the experiment, so callers can call `log_*` against it."""

    run_id: str
    experiment_id: str


@contextmanager
def start_run(cfg: DictConfig, oxen_meta: OxenMeta | None = None):
    """Start (and on exit, end) an mlflow run.

    Configures `tracking_uri` and `experiment` from `cfg.mlflow`, sets
    the dataset-identity tags from `oxen_meta` (REQ-014), the model
    architecture tag (NFR-008 filtering), and logs the resolved
    hyperparameters as run parameters.
    """
    if cfg.mlflow.tracking_uri:
        mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.mlflow.experiment)

    with mlflow.start_run() as run:
        tags = {
            TAG_MODEL_ARCHITECTURE: getattr(cfg.model, "name", "unknown"),
        }
        if oxen_meta is not None:
            tags[TAG_DATASET_OXEN_COMMIT] = oxen_meta.commit_hash
            tags[TAG_DATASET_DIRTY] = str(oxen_meta.dirty).lower()
        tags[TAG_DATASET_OXEN_DIR] = str(cfg.data.oxen_dir)
        mlflow.set_tags(tags)

        mlflow.log_params(_flat_params(cfg))
        yield RunHandle(run_id=run.info.run_id, experiment_id=run.info.experiment_id)


def log_batch(step: int, loss: float) -> None:
    """Per-batch hook — `train/loss` time series."""
    mlflow.log_metric("train/loss", loss, step=step)


def log_epoch(epoch: int, metrics: dict[str, float]) -> None:
    """Per-epoch hook — logs each key under `val/` (or as-is if it has a
    slash) so the UI groups them sensibly."""
    for k, v in metrics.items():
        mlflow.log_metric(_namespaced_metric_key(k), float(v), step=epoch)


def log_final(test_metrics: dict[str, float], artifact_dir: Path) -> None:
    """End-of-run: log final test metrics (each as a single-step series)
    and the artifact directory as a run artifact bundle."""
    for k, v in test_metrics.items():
        mlflow.log_metric(_namespaced_metric_key(k, default_ns="test"), float(v))
    if artifact_dir.exists():
        mlflow.log_artifacts(str(artifact_dir))


def build_callbacks() -> TrainingCallbacks:
    """A `TrainingCallbacks` whose hooks route to `log_batch` / `log_epoch`."""
    return TrainingCallbacks(on_batch_end=log_batch, on_epoch_end=log_epoch)


# ---------------------------------------------------------------------------- #
# Internals                                                                    #
# ---------------------------------------------------------------------------- #


def _flat_params(cfg: DictConfig, prefix: str = "") -> dict[str, str]:
    """Flatten a nested DictConfig into mlflow-friendly key:value strings."""
    from omegaconf import DictConfig as _DC
    from omegaconf import ListConfig

    out: dict[str, str] = {}
    for k, v in cfg.items():
        key = f"{prefix}{k}"
        if isinstance(v, _DC):
            out.update(_flat_params(v, prefix=f"{key}."))
        elif isinstance(v, ListConfig):
            out[key] = str(list(v))
        else:
            out[key] = str(v)
    return out


def _namespaced_metric_key(name: str, *, default_ns: str = "val") -> str:
    # mlflow allows only alphanumerics, _, -, ., space, : and / in metric
    # names. Our natural keys sometimes contain "@" (e.g. iou_f1@0.5);
    # translate to "_at_" so callers can keep the readable form upstream.
    safe = name.replace("@", "_at_")
    if "/" in safe:
        return safe
    return f"{default_ns}/{safe}"
