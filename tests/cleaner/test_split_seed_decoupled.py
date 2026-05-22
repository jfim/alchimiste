"""Verify `data.split_seed` is decoupled from the top-level `seed`.

`seed` should control training RNGs only (weight init, DataLoader
shuffle, dropout). `data.split_seed` should control the train/val/test
partition only. Setting them to different values lets multi-seed
averaging hold the split fixed while varying only the training
trajectory — which is what makes seed-averaged comparisons honest.

Default `data.split_seed: ${seed}` keeps the legacy behavior intact:
when only `seed` is set, the split matches what it used to be.
"""

from __future__ import annotations

import hashlib
import importlib
import json
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
from alchimiste.cleaner.data.split import make_splits
from alchimiste.cleaner.models.base import TrainingCallbacks
from alchimiste.cleaner.training.loop import train

CONFIGS_DIR = (Path(__file__).resolve().parents[2] / "configs").as_posix()
_STUB_MODULE_NAME = "alchimiste_test_split_seed_stub"


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
        callbacks.on_batch_end(0, 0.5)
        callbacks.on_epoch_end(0, {"val_loss": 0.4})

    def predict_token_probs(self, examples):
        return [[0.1 for _ in e.input_ids] for e in examples]

    def save(self, dst: Path) -> None:
        Path(dst).mkdir(parents=True, exist_ok=True)


def _register_stub() -> None:
    m = types.ModuleType(_STUB_MODULE_NAME)
    m.Tagger = _StubTagger
    sys.modules[_STUB_MODULE_NAME] = m
    importlib.invalidate_caches()


def _synth_oxen_tree(tmp_path: Path, n: int = 30) -> Path:
    stage_dir = tmp_path / "cleaning"
    blobs = stage_dir / "blobs"
    blobs.mkdir(parents=True)
    rows = []
    for i in range(n):
        body = f"Article {i}.".encode()
        sha = hashlib.sha256(body).hexdigest()
        (blobs / sha).write_bytes(body)
        rows.append({"item_id": f"a_{i:02d}", "content_sha256": sha, "discard_ranges": []})
    pl.DataFrame(rows).write_parquet(stage_dir / "rows.parquet")
    return tmp_path


def _compose(oxen_dir: Path, out_dir: Path, tracking_uri: str, *extra_overrides: str):
    if GlobalHydra.instance().is_initialized():
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
                "+model.name=stub_split_seed",
                f"mlflow.tracking_uri={tracking_uri}",
                f"mlflow.experiment=alchimiste_test_split_seed_{out_dir.name}",
                *extra_overrides,
            ],
            return_hydra_config=True,
        )
    cfg.hydra.runtime.output_dir = str(out_dir)
    HydraConfig.instance().set_config(cfg)
    return cfg


def _read_split(out_dir: Path) -> dict:
    return json.loads((out_dir / "splits.json").read_text())


@pytest.fixture
def env(tmp_path: Path) -> tuple[Path, str]:
    _register_stub()
    oxen_dir = _synth_oxen_tree(tmp_path / "oxen")
    tracking_uri = f"file://{tmp_path / 'mlruns'}"
    return oxen_dir, tracking_uri


def test_default_split_seed_matches_seed(env, tmp_path: Path) -> None:
    """When only `seed` is set, the split is identical to what the legacy
    behavior (`make_splits(seed=cfg.seed)`) produced."""
    oxen_dir, tracking = env
    out_dir = tmp_path / "out"
    cfg = _compose(oxen_dir, out_dir, tracking, "seed=42")
    user_cfg = OmegaConf.masked_copy(cfg, ["data", "model", "training", "eval", "mlflow", "seed"])
    train(user_cfg)

    actual = _read_split(out_dir)
    expected = make_splits(item_ids=[f"a_{i:02d}" for i in range(30)], seed=42)
    assert sorted(actual["train"]) == sorted(expected.train)
    assert sorted(actual["val"]) == sorted(expected.val)
    assert sorted(actual["test"]) == sorted(expected.test)


def test_split_seed_overrides_seed_for_partition_only(env, tmp_path: Path) -> None:
    """Setting `data.split_seed` independently of `seed` produces the
    split that `make_splits(seed=split_seed)` would — not `seed`'s split."""
    oxen_dir, tracking = env
    out_dir = tmp_path / "out"
    cfg = _compose(oxen_dir, out_dir, tracking, "seed=42", "data.split_seed=11")
    user_cfg = OmegaConf.masked_copy(cfg, ["data", "model", "training", "eval", "mlflow", "seed"])
    train(user_cfg)

    actual = _read_split(out_dir)
    from_split_seed = make_splits(item_ids=[f"a_{i:02d}" for i in range(30)], seed=11)
    from_top_seed = make_splits(item_ids=[f"a_{i:02d}" for i in range(30)], seed=42)
    # Must match the split_seed, not the top seed.
    assert sorted(actual["train"]) == sorted(from_split_seed.train)
    assert sorted(actual["train"]) != sorted(from_top_seed.train), (
        "If this fires, `seed` is still driving the split — decoupling regressed"
    )


def test_same_split_seed_with_different_top_seeds_gives_identical_split(
    env, tmp_path: Path
) -> None:
    """Two runs at different `seed` values but the same `data.split_seed`
    must produce the same partition. This is the case multi-seed
    averaging on a fixed split relies on."""
    oxen_dir, tracking = env
    out_dir_a = tmp_path / "out_a"
    out_dir_b = tmp_path / "out_b"

    user_keys = ["data", "model", "training", "eval", "mlflow", "seed"]
    cfg_a = _compose(oxen_dir, out_dir_a, tracking, "seed=11", "data.split_seed=17")
    train(OmegaConf.masked_copy(cfg_a, user_keys))

    cfg_b = _compose(oxen_dir, out_dir_b, tracking, "seed=42", "data.split_seed=17")
    train(OmegaConf.masked_copy(cfg_b, user_keys))

    split_a = _read_split(out_dir_a)
    split_b = _read_split(out_dir_b)
    assert sorted(split_a["train"]) == sorted(split_b["train"])
    assert sorted(split_a["val"]) == sorted(split_b["val"])
    assert sorted(split_a["test"]) == sorted(split_b["test"])
