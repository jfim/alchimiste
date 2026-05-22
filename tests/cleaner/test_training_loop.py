"""Integration test for `training.loop.train` (TASK-013).

Wires data -> tokenize -> fit -> predict -> RunResult using a stub
`TokenTagger` (no HF download). Confirms NFR-008: artifact paths are
under the Hydra `runtime.output_dir` and never reach into a shared
mutable location.
"""

from __future__ import annotations

import hashlib
import importlib
import sys
import types
from pathlib import Path

import polars as pl
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf

from alchimiste.cleaner.data.align import TokenizedExample
from alchimiste.cleaner.data.loader import LabeledArticle
from alchimiste.cleaner.models.base import TrainingCallbacks
from alchimiste.cleaner.training.loop import train

CONFIGS_DIR = (Path(__file__).resolve().parents[2] / "configs").as_posix()


# --------------------------------------------------------------------------- #
# Stub tagger as an importable module (training.loop uses importlib).         #
# --------------------------------------------------------------------------- #


_FIT_CALL_LOG: list[dict] = []


class _StubTagger:
    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.fit_called = False

    def tokenize(
        self,
        articles: list[LabeledArticle],
        max_seq_len: int,
    ) -> list[TokenizedExample]:
        return [
            TokenizedExample(
                item_id=a.item_id,
                input_ids=(1, 2, 3),
                codepoint_offset_mapping=((0, 0), (0, 1), (1, 2)),
                labels=(-100, 0, 0),
            )
            for a in articles
        ]

    def fit(
        self,
        train: list[TokenizedExample],
        val: list[TokenizedExample],
        cfg,
        callbacks: TrainingCallbacks,
    ) -> None:
        self.fit_called = True
        _FIT_CALL_LOG.append({"train_n": len(train), "val_n": len(val), "callbacks": callbacks})
        callbacks.on_batch_end(0, 0.0)
        callbacks.on_epoch_end(0, {"loss": 0.0})

    def predict_token_probs(self, examples: list[TokenizedExample]) -> list[list[float]]:
        return [[0.1] * len(e.input_ids) for e in examples]

    def save(self, dst: Path) -> None:
        dst.mkdir(parents=True, exist_ok=True)
        (dst / "marker").write_text("stub", encoding="utf-8")

    @classmethod
    def load(cls, src: Path) -> _StubTagger:
        (src / "marker").read_text(encoding="utf-8")
        return cls(cfg={})


_STUB_MODULE_NAME = "alchimiste_test_stub_tagger"


def _register_stub_module() -> None:
    module = types.ModuleType(_STUB_MODULE_NAME)
    module.Tagger = _StubTagger
    sys.modules[_STUB_MODULE_NAME] = module
    importlib.invalidate_caches()


def _synthesize_oxen_tree(tmp_path: Path, n_articles: int = 10) -> Path:
    stage_dir = tmp_path / "cleaning"
    blobs_dir = stage_dir / "blobs"
    blobs_dir.mkdir(parents=True)
    rows = []
    for i in range(n_articles):
        text = f"Article {i:03d}. Some boilerplate. Real content."
        body = text.encode("utf-8")
        sha = hashlib.sha256(body).hexdigest()
        (blobs_dir / sha).write_bytes(body)
        rows.append(
            {
                "item_id": f"art_{i:03d}",
                "content_sha256": sha,
                # No drop ranges — keeps the test about wiring, not labeling.
                "discard_ranges": [],
            }
        )
    pl.DataFrame(rows).write_parquet(stage_dir / "rows.parquet")
    return tmp_path


def _compose_test_cfg(oxen_dir: Path, hydra_output_dir: Path):
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=CONFIGS_DIR, version_base=None):
        cfg = compose(
            config_name="config",
            overrides=[
                f"data.oxen_dir={oxen_dir}",
                "~model",  # encoder_hf.yaml not yet present
                "+model.module=" + _STUB_MODULE_NAME,
            ],
            return_hydra_config=True,
        )
    # Manually set the runtime output_dir so HydraConfig.get() inside the
    # loop resolves it (compose doesn't normally populate this).
    cfg.hydra.runtime.output_dir = str(hydra_output_dir)
    HydraConfig.instance().set_config(cfg)
    return cfg


def test_train_wires_data_through_fit_to_run_result(tmp_path: Path) -> None:
    _register_stub_module()
    _FIT_CALL_LOG.clear()
    oxen_dir = _synthesize_oxen_tree(tmp_path / "oxen", n_articles=12)
    out_dir = tmp_path / "hydra_out"

    cfg = _compose_test_cfg(oxen_dir, out_dir)
    # Strip the hydra meta-config before handing to train; train only
    # reads the top-level keys.
    user_cfg = OmegaConf.masked_copy(cfg, ["data", "model", "training", "eval", "mlflow", "seed"])

    result = train(user_cfg)

    # RunResult fields populated.
    assert result.artifact_dir == out_dir
    assert result.train_size + result.val_size + result.test_size == 12
    # fit() was actually called.
    assert len(_FIT_CALL_LOG) == 1
    assert _FIT_CALL_LOG[0]["train_n"] == result.train_size
    assert _FIT_CALL_LOG[0]["val_n"] == result.val_size
    # Artifacts on disk.
    assert (out_dir / "splits.json").exists()
    assert (out_dir / "config.yaml").exists()
    assert (out_dir / "model" / "marker").exists()


def test_train_writes_only_under_hydra_output_dir(tmp_path: Path) -> None:
    """NFR-008: no shared mutable paths. Everything must land under
    cfg.hydra.runtime.output_dir."""
    _register_stub_module()
    _FIT_CALL_LOG.clear()
    oxen_dir = _synthesize_oxen_tree(tmp_path / "oxen", n_articles=6)
    out_dir = tmp_path / "isolated_run"
    cfg = _compose_test_cfg(oxen_dir, out_dir)
    user_cfg = OmegaConf.masked_copy(cfg, ["data", "model", "training", "eval", "mlflow", "seed"])

    train(user_cfg)

    # The output dir contains the run's artifacts.
    assert list(out_dir.iterdir()), "expected artifacts under output_dir"
    # No siblings under tmp_path other than oxen + the isolated run.
    siblings = {p.name for p in tmp_path.iterdir()}
    assert siblings == {"oxen", "isolated_run"}


def test_parallel_runs_get_disjoint_output_dirs(tmp_path: Path) -> None:
    """Two simultaneous train() calls with different Hydra output_dirs do not
    collide (NFR-008 acceptance, simplified — full parallel-subprocess test
    lives in TASK-028)."""
    _register_stub_module()
    _FIT_CALL_LOG.clear()
    oxen_dir = _synthesize_oxen_tree(tmp_path / "oxen", n_articles=8)

    out_a = tmp_path / "run_a"
    out_b = tmp_path / "run_b"

    cfg_a = _compose_test_cfg(oxen_dir, out_a)
    user_a = OmegaConf.masked_copy(cfg_a, ["data", "model", "training", "eval", "mlflow", "seed"])
    r_a = train(user_a)

    cfg_b = _compose_test_cfg(oxen_dir, out_b)
    user_b = OmegaConf.masked_copy(cfg_b, ["data", "model", "training", "eval", "mlflow", "seed"])
    r_b = train(user_b)

    assert r_a.artifact_dir != r_b.artifact_dir
    # Both runs wrote independently.
    assert (out_a / "splits.json").exists() and (out_b / "splits.json").exists()
