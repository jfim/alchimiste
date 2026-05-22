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
from hydra.core.hydra_config import HydraConfig

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

    with mlflow.start_run(run_name=_run_name(cfg)) as run:
        tags = {
            TAG_MODEL_ARCHITECTURE: getattr(cfg.model, "name", "unknown"),
        }
        if oxen_meta is not None:
            tags[TAG_DATASET_OXEN_COMMIT] = oxen_meta.commit_hash
            tags[TAG_DATASET_DIRTY] = str(oxen_meta.dirty).lower()
        tags[TAG_DATASET_OXEN_DIR] = str(cfg.data.oxen_dir)
        # MLflow surfaces `mlflow.note.content` as the run's "Description"
        # in the UI. Optional: empty string → unset so we don't clutter
        # runs that didn't bother to add a note.
        note = str(cfg.mlflow.get("note", "") or "").strip()
        if note:
            tags["mlflow.note.content"] = note
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
    """End-of-run: log final test metrics and the small browsable sidecars.

    The trained weights are *not* logged here — they ship as a registered
    pyfunc via `log_pyfunc_model`, which is the single source of truth.
    Logging the whole `artifact_dir` would duplicate the weights on the
    tracking server for no benefit.
    """
    for k, v in test_metrics.items():
        mlflow.log_metric(_namespaced_metric_key(k, default_ns="test"), float(v))
    for sidecar in _SIDECAR_FILES:
        path = artifact_dir / sidecar
        if path.exists():
            mlflow.log_artifact(str(path))


# Text-only files worth keeping at the run root for quick UI browsing.
# Model weights live in the registered pyfunc, not here.
_SIDECAR_FILES = (
    "config.yaml",
    "metrics.json",
    "threshold.json",
    "splits.json",
    "failures.jsonl",
)


# Soft warning threshold for the registered model footprint. Lives here
# so the training loop can call it without re-deriving the constant.
_ARTIFACT_SIZE_WARN_BYTES = 2 * 1024 * 1024 * 1024  # 2 GB (NFR-004)


def log_pyfunc_model(
    artifact_dir: Path,
    *,
    registered_model_name: str,
    threshold: float,
    threshold_iou: float,
    fell_back_to_max_precision: bool,
) -> None:
    """Bundle the training artifact + inference code into an mlflow pyfunc
    and register it under `registered_model_name` (TASK-025, REQ-011).

    The mlflow model contains:
      * the saved tagger under `artifact_root/model/`
      * the resolved Hydra config
      * the selected threshold + sweep + final metrics + failures
      * the alchimiste Python package (so `load_context` works with
        only the mlflow client installed at inference time)
    """
    import alchimiste

    package_root = Path(alchimiste.__file__).resolve().parent

    # NFR-004 soft warning.
    total_size = _dir_size_bytes(artifact_dir)
    if total_size > _ARTIFACT_SIZE_WARN_BYTES:
        import warnings

        warnings.warn(
            f"artifact_dir is {total_size / (1024**3):.2f} GB; mlflow registry "
            f"will accept it but consider whether the architecture is right-sized "
            f"(NFR-004 soft cap 2 GB).",
            stacklevel=2,
        )

    # Tag the active run with the chosen threshold.
    mlflow.set_tag(TAG_THRESHOLD_VALUE, f"{threshold:.6f}")
    mlflow.set_tag(TAG_THRESHOLD_IOU, f"{threshold_iou:.2f}")
    if fell_back_to_max_precision:
        mlflow.set_tag(TAG_THRESHOLD_FALLBACK, "true")

    # `artifacts` lets `CleanerModel.load_context` find the per-run
    # files without us baking absolute paths into the model.
    from alchimiste.cleaner.inference.pyfunc import CleanerModel

    # Override mlflow's auto-inferred env so the registered model stays
    # device-portable. By default mlflow snapshots whatever torch was
    # active at training time — a CUDA training run would otherwise
    # pin `torch==X.Y.Z+cu132`, forcing every CPU-only consumer to pull
    # a ~2.5 GB CUDA wheel they can't use. The trained weights load
    # fine into either build.
    mlflow.pyfunc.log_model(
        name="model",
        python_model=CleanerModel(),
        artifacts={"artifact_root": str(artifact_dir)},
        code_paths=[str(package_root)],
        registered_model_name=registered_model_name,
        pip_requirements=_pyfunc_pip_requirements(),
    )


def _pyfunc_pip_requirements() -> list[str]:
    """Pip-requirement strings to bundle with the registered pyfunc.

    Intentionally device-agnostic: pin only the upper-and-lower torch
    range that the inference code supports, not the local-variant tag
    (`+cpu` / `+cuXXX`). Other deps come from the inference path's
    imports and are listed here so mlflow doesn't fall back to env
    snapshotting and accidentally embed CUDA wheels.
    """
    return [
        "torch>=2.12,<3",
        "transformers>=5.9,<6",
        "tokenizers>=0.22",
        "numpy>=2.4",
        "omegaconf>=2.3",
    ]


def _dir_size_bytes(path: Path) -> int:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


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


def _run_name(cfg: DictConfig) -> str:
    """Readable run name so a sweep is scannable in the UI without
    drilling into tags.

    Format:
      * `<arch>-<suffix>` when `cfg.mlflow.run_name_suffix` is set
        (autoresearch passes a short slug per experiment, e.g. "ep6-clw10")
      * `<arch>-<timestamp>` otherwise — the Hydra run stamp keeps
        back-to-back runs with identical configs distinguishable.
    """
    arch = getattr(cfg.model, "name", "unknown")
    suffix = str(cfg.mlflow.get("run_name_suffix", "") or "").strip()
    if suffix:
        return f"{arch}-{suffix}"
    stamp = _hydra_run_stamp()
    return f"{arch}-{stamp}" if stamp else arch


def _hydra_run_stamp() -> str | None:
    """Basename of the Hydra runtime output dir (e.g. "03-29-46"); None
    when called outside a Hydra-managed run (unit tests)."""
    try:
        output_dir = HydraConfig.get().runtime.output_dir
    except ValueError:
        return None
    p = Path(output_dir)
    return f"{p.parent.name}_{p.name}"


def _namespaced_metric_key(name: str, *, default_ns: str = "val") -> str:
    # mlflow allows only alphanumerics, _, -, ., space, : and / in metric
    # names. Our natural keys sometimes contain "@" (e.g. iou_f1@0.5);
    # translate to "_at_" so callers can keep the readable form upstream.
    safe = name.replace("@", "_at_")
    if "/" in safe:
        return safe
    return f"{default_ns}/{safe}"
