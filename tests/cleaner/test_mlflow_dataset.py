"""Verify oxen-backed dataset is logged as an MLflow `MetaDataset`.

This surfaces the dataset in MLflow's Datasets column (not just as tags)
with a name like `cleaning-178fc8-n-193` so the UI shows at a glance what
went into the experiment.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from alchimiste.cleaner.data.oxen_meta import OxenMeta
from alchimiste.cleaner.training import mlflow_io
from alchimiste.cleaner.training.mlflow_io import (
    OxenDatasetSource,
    _build_dataset,
    _count_articles,
)


def _make_oxen_tree(root: Path, *, n_blobs: int = 0, with_rows: bool = False) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    if n_blobs:
        blobs = root / "blobs"
        blobs.mkdir()
        for i in range(n_blobs):
            (blobs / f"hash{i:04d}").write_text("x")
    if with_rows:
        import pyarrow as pa
        import pyarrow.parquet as pq

        pq.write_table(pa.table({"id": list(range(7))}), root / "rows.parquet")
    return root


def test_count_articles_from_blobs(tmp_path: Path) -> None:
    tree = _make_oxen_tree(tmp_path / "cleaning", n_blobs=5)
    assert _count_articles(tree) == 5


def test_count_articles_falls_back_to_rows(tmp_path: Path) -> None:
    tree = _make_oxen_tree(tmp_path / "cleaning", with_rows=True)
    assert _count_articles(tree) == 7


def test_count_articles_none_when_neither(tmp_path: Path) -> None:
    tree = tmp_path / "cleaning"
    tree.mkdir()
    assert _count_articles(tree) is None


def test_build_dataset_clean(tmp_path: Path) -> None:
    tree = _make_oxen_tree(tmp_path / "cleaning", n_blobs=193)
    meta = OxenMeta(commit_hash="178fc8abcd12", dirty=False)
    ds = _build_dataset(meta, tree)
    assert ds.name == "cleaning-178fc8-n-193"


def test_build_dataset_dirty_suffix(tmp_path: Path) -> None:
    tree = _make_oxen_tree(tmp_path / "cleaning", n_blobs=193)
    meta = OxenMeta(commit_hash="178fc8abcd12", dirty=True)
    ds = _build_dataset(meta, tree)
    assert ds.name == "cleaning-178fc8-n-193-dirty"


def test_build_dataset_unknown_count(tmp_path: Path) -> None:
    tree = tmp_path / "cleaning"
    tree.mkdir()
    meta = OxenMeta(commit_hash="aabbccddeeff", dirty=False)
    ds = _build_dataset(meta, tree)
    assert ds.name == "cleaning-aabbcc-n-?"


def test_oxen_source_round_trip() -> None:
    src = OxenDatasetSource(oxen_dir="/tmp/cleaning", commit_hash="abc123")
    restored = OxenDatasetSource.from_dict(src.to_dict())
    assert restored.oxen_dir == "/tmp/cleaning"
    assert restored.commit_hash == "abc123"
    assert OxenDatasetSource._get_source_type() == "oxen"


def test_oxen_source_load_unsupported() -> None:
    src = OxenDatasetSource(oxen_dir="/tmp/x", commit_hash="abc")
    with pytest.raises(NotImplementedError):
        src.load()


# ---------- start_run integration ----------


class _DummyRun:
    class info:
        run_id = "r"
        experiment_id = "e"


@contextmanager
def _dummy_run():
    yield _DummyRun()


@pytest.fixture
def patch_mlflow(monkeypatch):
    inputs: list = []

    monkeypatch.setattr(mlflow_io.mlflow, "set_tracking_uri", lambda *a, **kw: None)
    monkeypatch.setattr(mlflow_io.mlflow, "set_experiment", lambda *a, **kw: None)
    monkeypatch.setattr(mlflow_io.mlflow, "start_run", lambda *a, **kw: _dummy_run())
    monkeypatch.setattr(mlflow_io.mlflow, "set_tags", lambda *a, **kw: None)
    monkeypatch.setattr(mlflow_io.mlflow, "log_params", lambda *a, **kw: None)
    monkeypatch.setattr(mlflow_io.mlflow, "log_input", lambda d: inputs.append(d))
    return inputs


def _cfg(oxen_dir: str):
    return OmegaConf.create(
        {
            "model": {"name": "encoder_hf"},
            "data": {"oxen_dir": oxen_dir},
            "mlflow": {
                "tracking_uri": "",
                "experiment": "test",
                "run_name_suffix": "",
                "note": "",
                "autoresearch": "",
            },
        }
    )


def test_start_run_logs_dataset_when_meta_given(patch_mlflow, tmp_path: Path) -> None:
    tree = _make_oxen_tree(tmp_path / "cleaning", n_blobs=4)
    meta = OxenMeta(commit_hash="deadbeef0000", dirty=False)
    with mlflow_io.start_run(_cfg(str(tree)), oxen_meta=meta):
        pass
    assert len(patch_mlflow) == 1
    assert patch_mlflow[0].name == "cleaning-deadbe-n-4"


def test_start_run_no_dataset_when_meta_missing(patch_mlflow, tmp_path: Path) -> None:
    with mlflow_io.start_run(_cfg(str(tmp_path / "cleaning"))):
        pass
    assert patch_mlflow == []
