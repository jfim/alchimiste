"""Verify the `mlflow.autoresearch` knob lands as the right tag.

Empty → unset (manual runs unmarked). Non-empty → tag value verbatim so
the MLflow UI can filter by presence (any non-empty) and equality.
"""

from __future__ import annotations

from contextlib import contextmanager

import pytest
from omegaconf import OmegaConf

from alchimiste.cleaner.training import mlflow_io
from alchimiste.cleaner.training.mlflow_io import TAG_AUTORESEARCH


def _cfg(**mlflow_overrides):
    return OmegaConf.create(
        {
            "model": {"name": "encoder_hf"},
            "data": {"oxen_dir": "/tmp/x"},
            "mlflow": {
                "tracking_uri": "",
                "experiment": "test",
                "run_name_suffix": "",
                "note": "",
                "autoresearch": "",
                **mlflow_overrides,
            },
        }
    )


class _DummyRun:
    class info:
        run_id = "r"
        experiment_id = "e"


@contextmanager
def _dummy_run():
    yield _DummyRun()


@pytest.fixture
def capture_tags(monkeypatch):
    """Patch out everything that touches the network / global state and
    capture the tags dict that `start_run` would set."""
    captured: dict[str, str] = {}

    def fake_set_tags(tags):
        captured.update(tags)

    monkeypatch.setattr(mlflow_io.mlflow, "set_tracking_uri", lambda *a, **kw: None)
    monkeypatch.setattr(mlflow_io.mlflow, "set_experiment", lambda *a, **kw: None)
    monkeypatch.setattr(mlflow_io.mlflow, "start_run", lambda *a, **kw: _dummy_run())
    monkeypatch.setattr(mlflow_io.mlflow, "set_tags", fake_set_tags)
    monkeypatch.setattr(mlflow_io.mlflow, "log_params", lambda *a, **kw: None)
    return captured


def test_autoresearch_tag_present_when_set(capture_tags):
    with mlflow_io.start_run(_cfg(autoresearch="batch-1")):
        pass
    assert capture_tags[TAG_AUTORESEARCH] == "batch-1"


def test_autoresearch_tag_absent_when_empty(capture_tags):
    with mlflow_io.start_run(_cfg(autoresearch="")):
        pass
    assert TAG_AUTORESEARCH not in capture_tags


def test_autoresearch_tag_treats_whitespace_as_empty(capture_tags):
    with mlflow_io.start_run(_cfg(autoresearch="   ")):
        pass
    assert TAG_AUTORESEARCH not in capture_tags


def test_autoresearch_tag_constant_value() -> None:
    # Pin the exact key shape so an accidental rename is loud.
    assert TAG_AUTORESEARCH == "alchimiste.autoresearch"
