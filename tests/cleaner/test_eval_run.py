"""Tests for `eval.run.evaluate` and the `just eval` entrypoint (TASK-022, NFR-006)."""

from __future__ import annotations

import hashlib
import importlib
import json
import sys
import types
from pathlib import Path

import polars as pl
from omegaconf import OmegaConf

from alchimiste.cleaner.data.align import TokenizedExample
from alchimiste.cleaner.data.split import make_splits
from alchimiste.cleaner.eval.run import evaluate

_STUB_MODULE_NAME = "alchimiste_test_eval_stub"


class _StubTagger:
    """Deterministic stub for testing — always returns the same probs."""

    def __init__(self, cfg) -> None:
        self.cfg = cfg

    def tokenize(self, articles, max_seq_len: int):
        return [
            TokenizedExample(
                item_id=a.item_id,
                input_ids=(1, 2, 3),
                codepoint_offset_mapping=((0, 0), (0, 5), (5, 10)),
                labels=(-100, 0, 0),
            )
            for a in articles
        ]

    def fit(self, train, val, cfg, callbacks):
        pass

    def predict_token_probs(self, examples):
        # Deterministic: every token gets probability 0.1 (below threshold).
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


def _synth_artifact(tmp_path: Path, n: int = 12) -> Path:
    """Build a minimal artifact + paired oxen tree on disk."""
    oxen_dir = tmp_path / "oxen"
    stage_dir = oxen_dir / "cleaning"
    blobs = stage_dir / "blobs"
    blobs.mkdir(parents=True)
    rows = []
    for i in range(n):
        body = f"Article {i}.".encode()
        sha = hashlib.sha256(body).hexdigest()
        (blobs / sha).write_bytes(body)
        rows.append({"item_id": f"a_{i}", "content_sha256": sha, "discard_ranges": []})
    pl.DataFrame(rows).write_parquet(stage_dir / "rows.parquet")

    artifact_dir = tmp_path / "artifact"
    artifact_dir.mkdir()
    (artifact_dir / "model").mkdir()
    (artifact_dir / "model" / "marker").write_text("stub")

    cfg = OmegaConf.create(
        {
            "data": {
                "oxen_dir": str(oxen_dir),
                "stage": "cleaning",
                "allow_dirty": True,
                "require_nfc": True,
                "range_units": "byte",
                "max_seq_len": 64,
            },
            "model": {
                "name": "stub_for_eval_test",
                "module": _STUB_MODULE_NAME,
            },
            "training": {
                "batch_size": 4,
                "learning_rate": 1.0e-4,
                "epochs": 1,
                "class_weight_drop": 1.0,
                "device": "cpu",
            },
            "eval": {
                "primary_iou": 0.5,
                "report_iou": [0.5, 0.9, 1.0],
                "precision_floor": 0.85,
                "threshold_sweep": {"min": 0.3, "max": 0.9, "step": 0.1},
                "excessive_ranges_threshold": 6,
            },
            "seed": 17,
        }
    )
    OmegaConf.save(cfg, artifact_dir / "config.yaml")

    splits = make_splits(item_ids=[f"a_{i}" for i in range(n)], seed=17)
    splits.write_json(artifact_dir / "splits.json")
    return artifact_dir


def test_evaluate_two_consecutive_invocations_yield_identical_metrics_json(
    tmp_path: Path,
) -> None:
    """NFR-006 acceptance: byte-identical metrics.json across re-runs."""
    _register_stub()
    artifact_dir = _synth_artifact(tmp_path)

    evaluate(artifact_dir)
    first = (artifact_dir / "metrics.json").read_bytes()

    evaluate(artifact_dir)
    second = (artifact_dir / "metrics.json").read_bytes()

    assert first == second, "metrics.json changed between consecutive eval runs"


def test_evaluate_writes_all_expected_artifacts(tmp_path: Path) -> None:
    _register_stub()
    artifact_dir = _synth_artifact(tmp_path)
    evaluate(artifact_dir)
    for filename in ("metrics.json", "threshold.json", "failures.jsonl"):
        assert (artifact_dir / filename).exists(), f"missing {filename}"


def test_evaluate_returns_expected_metric_keys(tmp_path: Path) -> None:
    _register_stub()
    artifact_dir = _synth_artifact(tmp_path)
    metrics = evaluate(artifact_dir)
    # REQ-003 keys present.
    assert "test/clean_articles_kept_clean_fraction" in metrics
    assert "test/excessive_ranges_fraction" in metrics
    # IoU keys for the three thresholds.
    for tau_label in ("0.5", "0.9", "1"):
        assert f"test/iou_f1_at_{tau_label}" in metrics


def test_evaluate_with_override_oxen_dir(tmp_path: Path) -> None:
    _register_stub()
    artifact_dir = _synth_artifact(tmp_path)
    # Build a different oxen tree and override.
    alt_oxen = tmp_path / "alt_oxen"
    stage_dir = alt_oxen / "cleaning"
    blobs = stage_dir / "blobs"
    blobs.mkdir(parents=True)
    body = b"Different article."
    sha = hashlib.sha256(body).hexdigest()
    (blobs / sha).write_bytes(body)
    pl.DataFrame(
        [
            {
                "item_id": "a_0",  # match one of the trained ids so it lands in some split
                "content_sha256": sha,
                "discard_ranges": [],
            }
        ]
    ).write_parquet(stage_dir / "rows.parquet")

    metrics = evaluate(artifact_dir, override_oxen_dir=alt_oxen)
    assert "test/clean_articles_kept_clean_fraction" in metrics
    # Sanity: metrics.json still exists post-override.
    assert json.loads((artifact_dir / "metrics.json").read_text())
