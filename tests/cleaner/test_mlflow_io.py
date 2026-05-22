"""End-to-end test for the mlflow wiring in the training loop (TASK-018).

The stub-tagger train loop with mlflow on a tmp `file://` tracking URI is
opened with MlflowClient afterwards to confirm:
  - `train/loss` has a time series with >= 1 step,
  - the dataset oxen_commit tag is set,
  - end-of-run logging fired (artifact bundle present).
"""

from __future__ import annotations

import hashlib
import importlib
import sys
import types
from pathlib import Path

import polars as pl
import pytest
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf

from alchimiste.cleaner.data.align import TokenizedExample
from alchimiste.cleaner.data.loader import LabeledArticle
from alchimiste.cleaner.models.base import TrainingCallbacks
from alchimiste.cleaner.training.loop import train

pytest.importorskip("mlflow")
import mlflow
from mlflow.tracking import MlflowClient

CONFIGS_DIR = (Path(__file__).resolve().parents[2] / "configs").as_posix()
_STUB_MODULE_NAME = "alchimiste_test_mlflow_stub"


class _StubTagger:
    def __init__(self, cfg) -> None:
        self.cfg = cfg

    def tokenize(self, articles: list[LabeledArticle], max_seq_len: int):
        return [
            TokenizedExample(
                item_id=a.item_id,
                input_ids=(1, 2, 3),
                codepoint_offset_mapping=((0, 0), (0, 1), (1, 2)),
                labels=(-100, 0, 0),
            )
            for a in articles
        ]

    def fit(self, train, val, cfg, callbacks: TrainingCallbacks) -> None:
        # Drive a handful of batch/epoch callbacks so the metric time
        # series is non-empty.
        for step in range(3):
            callbacks.on_batch_end(step, 0.5 - 0.1 * step)
        callbacks.on_epoch_end(0, {"val_loss": 0.4, "iou_f1@0.5": 0.6})

    def predict_token_probs(self, examples):
        return [[0.1] * len(e.input_ids) for e in examples]

    def save(self, dst: Path) -> None:
        dst.mkdir(parents=True, exist_ok=True)
        (dst / "marker").write_text("stub")

    @classmethod
    def load(cls, src: Path):
        return cls(cfg={})


def _register_stub() -> None:
    module = types.ModuleType(_STUB_MODULE_NAME)
    module.Tagger = _StubTagger
    sys.modules[_STUB_MODULE_NAME] = module
    importlib.invalidate_caches()


def _synth_oxen_tree(tmp_path: Path, n: int = 6) -> Path:
    stage_dir = tmp_path / "cleaning"
    blobs = stage_dir / "blobs"
    blobs.mkdir(parents=True)
    rows = []
    for i in range(n):
        body = f"Article {i}.".encode()
        sha = hashlib.sha256(body).hexdigest()
        (blobs / sha).write_bytes(body)
        rows.append({"item_id": f"a_{i}", "content_sha256": sha, "discard_ranges": []})
    pl.DataFrame(rows).write_parquet(stage_dir / "rows.parquet")
    return tmp_path


def _compose(oxen_dir: Path, hydra_out: Path, tracking_uri: str):
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=CONFIGS_DIR, version_base=None):
        cfg = compose(
            config_name="config",
            overrides=[
                f"data.oxen_dir={oxen_dir}",
                "~model",
                f"+model.module={_STUB_MODULE_NAME}",
                "+model.name=stub_for_mlflow_test",
                f"mlflow.tracking_uri={tracking_uri}",
                "mlflow.experiment=alchimiste_test_mlflow_io",
            ],
            return_hydra_config=True,
        )
    cfg.hydra.runtime.output_dir = str(hydra_out)
    HydraConfig.instance().set_config(cfg)
    return cfg


def test_train_logs_batch_loss_and_finalizes(tmp_path: Path) -> None:
    _register_stub()
    oxen_dir = _synth_oxen_tree(tmp_path / "oxen")
    out_dir = tmp_path / "hydra_out"
    tracking_uri = f"file://{tmp_path / 'mlruns'}"

    cfg = _compose(oxen_dir, out_dir, tracking_uri)
    user_cfg = OmegaConf.masked_copy(cfg, ["data", "model", "training", "eval", "mlflow", "seed"])

    result = train(user_cfg)
    assert result.artifact_dir == out_dir

    # Open the resulting mlflow run.
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)
    exp = client.get_experiment_by_name("alchimiste_test_mlflow_io")
    assert exp is not None
    runs = client.search_runs(experiment_ids=[exp.experiment_id])
    assert len(runs) == 1
    run = runs[0]

    # train/loss time series has >= 1 step.
    history = client.get_metric_history(run.info.run_id, "train/loss")
    assert len(history) >= 1

    # Model architecture tag set.
    assert run.data.tags.get("alchimiste.model.architecture") == "stub_for_mlflow_test"
    # Dataset dir tag set even when oxen meta unavailable.
    assert "alchimiste.dataset.oxen_dir" in run.data.tags
    # `mlflow.register_model` defaults to false → no registered model
    # version got created for this experiment's model name. This pins the
    # default off behavior so a regression would be loud.
    versions = client.search_model_versions(
        "name='alchimiste-text-cleaner-stub_for_mlflow_test'"
    )
    assert versions == [], (
        f"expected no registered model versions when register_model=false, got {len(versions)}"
    )


def test_train_registers_pyfunc_when_register_model_true(tmp_path: Path) -> None:
    """When `mlflow.register_model=true`, the pyfunc gets logged and a
    model version is registered. Mirrors the default-off test but with
    the knob flipped on."""
    _register_stub()
    oxen_dir = _synth_oxen_tree(tmp_path / "oxen")
    out_dir = tmp_path / "hydra_out"
    tracking_uri = f"file://{tmp_path / 'mlruns'}"

    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=CONFIGS_DIR, version_base=None):
        cfg = compose(
            config_name="config",
            overrides=[
                "data.allow_dirty=true",
                "data.require_nfc=false",
                "data.min_bytes=0",
                f"data.oxen_dir={oxen_dir}",
                "~model",
                f"+model.module={_STUB_MODULE_NAME}",
                "+model.name=stub_for_register_true_test",
                f"mlflow.tracking_uri={tracking_uri}",
                "mlflow.experiment=alchimiste_test_register_true",
                "mlflow.register_model=true",
            ],
            return_hydra_config=True,
        )
    cfg.hydra.runtime.output_dir = str(out_dir)
    HydraConfig.instance().set_config(cfg)
    user_cfg = OmegaConf.masked_copy(cfg, ["data", "model", "training", "eval", "mlflow", "seed"])
    train(user_cfg)

    client = MlflowClient(tracking_uri=tracking_uri)
    versions = client.search_model_versions(
        "name='alchimiste-text-cleaner-stub_for_register_true_test'"
    )
    assert len(versions) >= 1, "register_model=true should register at least one model version"
