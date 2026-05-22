"""Tests for `data.label_stats` (TASK-011)."""

from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path

import polars as pl

from alchimiste.cleaner.data.label_stats import compute_stats, render_stats
from alchimiste.cleaner.data.loader import LabeledArticle


def _article(item_id: str, text: str, ranges: list[tuple[int, int]]) -> LabeledArticle:
    return LabeledArticle(
        item_id=item_id,
        content_sha256="0" * 64,
        markdown_text=text,
        discard_ranges=tuple(ranges),
    )


def test_compute_stats_on_mixed_fixture() -> None:
    articles = [
        _article("a", "Hello world.", []),  # clean
        _article("b", "Drop me out", [(0, 5)]),
        _article("c", "AAA BBB CCC DDD", [(4, 7), (12, 15)]),
        _article("d", "BBB", [(0, 3)]),
    ]
    stats = compute_stats(articles)
    assert stats["total_articles"] == 4
    assert stats["clean_fraction"] == 0.25
    rc = stats["range_counts"]
    assert rc["min"] == 0
    assert rc["max"] == 2
    assert rc["mean"] == 1.0  # (0+1+2+1)/4
    rl = stats["range_lengths"]
    # Lengths: [5, 3, 3, 3]
    assert rl["min"] == 3
    assert rl["max"] == 5
    assert rl["mean"] == 3.5


def test_compute_stats_on_empty_corpus() -> None:
    stats = compute_stats([])
    assert stats["total_articles"] == 0
    assert stats["clean_fraction"] == 0.0


def test_compute_stats_all_clean_means_zero_range_lengths() -> None:
    articles = [_article("a", "x", []), _article("b", "y", [])]
    stats = compute_stats(articles)
    assert stats["clean_fraction"] == 1.0
    rl = stats["range_lengths"]
    assert rl["min"] == 0
    assert rl["max"] == 0  # no ranges to measure


def test_render_stats_is_human_readable() -> None:
    stats = compute_stats(
        [
            _article("a", "abc", [(0, 2)]),
            _article("b", "xyz", []),
        ]
    )
    text = render_stats(stats)
    # Keys present, numbers grep-able.
    assert "total_articles:    2" in text
    assert "clean_fraction:    0.500" in text
    assert "range_counts:" in text
    assert "range_lengths(cp):" in text


def test_label_stats_cli_runs_against_synthesized_oxen_tree(tmp_path: Path) -> None:
    """End-to-end: synthesize an oxen tree and invoke `python -m label_stats`."""
    text = "Hello world. Drop me. Goodbye."
    body = text.encode("utf-8")
    sha = hashlib.sha256(body).hexdigest()
    s = text.index("Drop")
    e = text.index("Goodbye")

    stage_dir = tmp_path / "cleaning"
    blobs_dir = stage_dir / "blobs"
    blobs_dir.mkdir(parents=True)
    (blobs_dir / sha).write_bytes(body)
    pl.DataFrame(
        [{"item_id": "a", "content_sha256": sha, "discard_ranges": [[s, e]]}]
    ).write_parquet(stage_dir / "rows.parquet")

    # Use Hydra overrides to point at the synthesized tree. The default
    # config pulls in `model=encoder_hf` which lands in TASK-014 — drop
    # it here since label-stats doesn't need a model.
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "alchimiste.cleaner.data.label_stats",
            f"data.oxen_dir={tmp_path}",
            "~model",
            f"hydra.run.dir={tmp_path}/_hydra_run",
            "hydra.output_subdir=null",
            "hydra/job_logging=disabled",
            "hydra/hydra_logging=disabled",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, (
        f"label-stats exited {result.returncode}\nstdout:{result.stdout}\nstderr:{result.stderr}"
    )
    assert "total_articles:    1" in result.stdout
    assert "clean_fraction:    0.000" in result.stdout
