"""Focused unit tests for `_run_name` and `note` plumbing in `mlflow_io`.

Heavier end-to-end coverage lives in `test_mlflow_io.py`; here we just
pin the run-name format and the empty-string fallbacks so future edits
don't silently regress them.
"""

from __future__ import annotations

from omegaconf import OmegaConf

from alchimiste.cleaner.training.mlflow_io import _run_name


def _cfg(**mlflow_overrides):
    return OmegaConf.create(
        {
            "model": {"name": "encoder_hf"},
            "mlflow": {"run_name_suffix": "", "note": "", **mlflow_overrides},
        }
    )


def test_run_name_uses_suffix_when_provided(monkeypatch):
    # Suffix wins over the Hydra stamp.
    monkeypatch.setattr(
        "alchimiste.cleaner.training.mlflow_io._hydra_run_stamp",
        lambda: "2026-05-22_05-39-27",
    )
    assert _run_name(_cfg(run_name_suffix="ep6-clw10")) == "encoder_hf-ep6-clw10"


def test_run_name_falls_back_to_hydra_stamp(monkeypatch):
    monkeypatch.setattr(
        "alchimiste.cleaner.training.mlflow_io._hydra_run_stamp",
        lambda: "2026-05-22_05-39-27",
    )
    assert _run_name(_cfg()) == "encoder_hf-2026-05-22_05-39-27"


def test_run_name_treats_whitespace_suffix_as_empty(monkeypatch):
    monkeypatch.setattr(
        "alchimiste.cleaner.training.mlflow_io._hydra_run_stamp",
        lambda: "stamp",
    )
    assert _run_name(_cfg(run_name_suffix="   ")) == "encoder_hf-stamp"


def test_run_name_no_seed_in_name(monkeypatch):
    # Regression: earlier format embedded `seed<N>` even though the seed
    # isn't part of the search space. Confirm it's gone.
    monkeypatch.setattr(
        "alchimiste.cleaner.training.mlflow_io._hydra_run_stamp",
        lambda: "stamp",
    )
    cfg = _cfg(run_name_suffix="ep6")
    name = _run_name(cfg)
    assert "seed" not in name


def test_run_name_without_hydra_stamp_returns_bare_arch(monkeypatch):
    monkeypatch.setattr(
        "alchimiste.cleaner.training.mlflow_io._hydra_run_stamp",
        lambda: None,
    )
    assert _run_name(_cfg()) == "encoder_hf"
