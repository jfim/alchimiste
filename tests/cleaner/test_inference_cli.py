"""Tests for `inference.pyfunc.CleanerModel` + the CLI (TASK-024, TASK-026, TASK-027)."""

from __future__ import annotations

import json
import subprocess
import sys
import unicodedata
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")

from omegaconf import OmegaConf

from alchimiste.cleaner.inference.pyfunc import CleanerModel, _LocalContext, predict_text


@pytest.fixture(scope="module")
def artifact_dir(tmp_path_factory) -> Path:
    """Train a tiny encoder_hf tagger once and reuse the artifact across tests."""
    from alchimiste.cleaner.data.loader import LabeledArticle
    from alchimiste.cleaner.models import encoder_hf
    from alchimiste.cleaner.models.base import TrainingCallbacks

    cfg = OmegaConf.create(
        {
            "name": "encoder_hf",
            "module": "alchimiste.cleaner.models.encoder_hf",
            "hf_model_name": "distilbert-base-uncased",
            "_training": {
                "batch_size": 2,
                "learning_rate": 1.0e-4,
                "epochs": 1,
                "class_weight_drop": 1.0,
                "device": "cpu",
            },
        }
    )
    try:
        tagger = encoder_hf.Tagger(cfg)
    except Exception as e:
        pytest.skip(f"could not load DistilBERT: {e}")

    arts = [
        LabeledArticle(
            item_id=f"a_{i}",
            content_sha256="0" * 64,
            markdown_text=f"Article {i}. Content here.",
            discard_ranges=(),
        )
        for i in range(4)
    ]
    ex = tagger.tokenize(arts, max_seq_len=32)
    tagger.fit(ex, ex[:1], cfg, TrainingCallbacks())

    artifact_dir = tmp_path_factory.mktemp("artifact")
    tagger.save(artifact_dir / "model")

    full_cfg = OmegaConf.create(
        {
            "data": {
                "oxen_dir": "n/a",
                "stage": "cleaning",
                "allow_dirty": True,
                "require_nfc": True,
                "range_units": "byte",
                "max_seq_len": 32,
            },
            "model": OmegaConf.to_container(cfg, resolve=True),
            "training": cfg._training,
            "eval": {
                "primary_iou": 0.5,
                "report_iou": [0.5],
                "precision_floor": 0.85,
                "threshold_sweep": {"min": 0.3, "max": 0.9, "step": 0.1},
                "excessive_ranges_threshold": 6,
            },
            "seed": 17,
        }
    )
    OmegaConf.save(full_cfg, artifact_dir / "config.yaml")

    (artifact_dir / "threshold.json").write_text(
        json.dumps(
            {
                "threshold": 0.5,
                "iou_metric": 0.5,
                "precision_at_chosen": 0.0,
                "recall_at_chosen": 0.0,
                "f1_at_chosen": 0.0,
                "fell_back_to_max_precision": True,
                "sweep": [],
            }
        )
        + "\n"
    )
    return artifact_dir


def test_predict_returns_drop_ranges_dict(artifact_dir: Path) -> None:
    result = predict_text(artifact_dir, "Hello world.")
    assert "drop_ranges" in result
    assert isinstance(result["drop_ranges"], list)
    for r in result["drop_ranges"]:
        assert isinstance(r, list) and len(r) == 2


def test_pyfunc_single_vs_list_input(artifact_dir: Path) -> None:
    model = CleanerModel()
    model.load_context(_LocalContext(artifact_dir))  # type: ignore[arg-type]
    single = model.predict(None, "Hello", None)  # type: ignore[arg-type]
    multi = model.predict(None, ["Hello", "Hi"], None)  # type: ignore[arg-type]
    assert isinstance(single, dict)
    assert isinstance(multi, list) and len(multi) == 2


def test_predict_defensively_nfc_normalizes(artifact_dir: Path) -> None:
    """REQ-013: NFD input should produce the same output as NFC."""
    nfc = "Café"
    nfd = unicodedata.normalize("NFD", nfc)
    assert nfc != nfd  # sanity
    out_nfc = predict_text(artifact_dir, nfc)
    out_nfd = predict_text(artifact_dir, nfd)
    assert out_nfc == out_nfd


def test_cli_emits_valid_json(artifact_dir: Path) -> None:
    """End-to-end CLI: stdin -> JSON stdout."""
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "alchimiste.cleaner.inference.cli",
            f"--artifact={artifact_dir}",
        ],
        input="Hello world.",
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, f"CLI exited {result.returncode}\n{result.stderr}"
    payload = json.loads(result.stdout.strip())
    assert "drop_ranges" in payload


def test_cli_bench_reports_elapsed(artifact_dir: Path) -> None:
    """TASK-027: --bench prints elapsed wall time to stderr."""
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "alchimiste.cleaner.inference.cli",
            f"--artifact={artifact_dir}",
            "--bench",
        ],
        input="Brief input.",
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, f"bench exited {result.returncode}\n{result.stderr}"
    assert "elapsed_seconds=" in result.stderr
