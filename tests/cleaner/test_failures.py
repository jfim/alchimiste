"""Tests for `eval.failures` (TASK-019, REQ-010)."""

from __future__ import annotations

import json
from pathlib import Path

from alchimiste.cleaner.eval.failures import ArticleResult, write_failures


def _r(item_id: str, text: str, true_, pred) -> ArticleResult:
    return ArticleResult(
        item_id=item_id,
        content_sha256="0" * 64,
        text=text,
        true_drop_ranges=tuple(true_),
        pred_drop_ranges=tuple(pred),
    )


def _read_jsonl(p: Path) -> list[dict]:
    return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines()]


def test_perfect_predictions_yield_empty_failures(tmp_path: Path) -> None:
    results = [_r("a", "hello world", [(0, 5)], [(0, 5)])]
    out = tmp_path / "failures.jsonl"
    n = write_failures(results, out)
    assert n == 0
    assert out.read_text() == ""


def test_extra_predicted_range_is_a_false_positive(tmp_path: Path) -> None:
    results = [_r("a", "abcdefgh", [(0, 3)], [(0, 3), (5, 8)])]
    out = tmp_path / "failures.jsonl"
    n = write_failures(results, out)
    assert n == 1
    [row] = _read_jsonl(out)
    assert row["false_positive_ranges"] == [[5, 8]]
    assert row["false_negative_ranges"] == []


def test_missing_predicted_range_is_a_false_negative(tmp_path: Path) -> None:
    results = [_r("a", "abcdefgh", [(0, 3), (5, 8)], [(0, 3)])]
    out = tmp_path / "failures.jsonl"
    write_failures(results, out)
    [row] = _read_jsonl(out)
    assert row["false_positive_ranges"] == []
    assert row["false_negative_ranges"] == [[5, 8]]


def test_pred_with_low_iou_is_both_fp_and_fn(tmp_path: Path) -> None:
    # truth=(0,10), pred=(8,18) -> IoU = 2/18 < 0.5
    results = [_r("a", "x" * 30, [(0, 10)], [(8, 18)])]
    out = tmp_path / "failures.jsonl"
    write_failures(results, out)
    [row] = _read_jsonl(out)
    assert row["false_positive_ranges"] == [[8, 18]]
    assert row["false_negative_ranges"] == [[0, 10]]


def test_kept_diff_contains_unified_diff_markers(tmp_path: Path) -> None:
    results = [_r("a", "Keep. DROP. Keep.", [(6, 11)], [])]
    out = tmp_path / "failures.jsonl"
    write_failures(results, out)
    [row] = _read_jsonl(out)
    assert "true_kept" in row["kept_diff"]
    assert "pred_kept" in row["kept_diff"]


def test_include_passes_writes_clean_rows(tmp_path: Path) -> None:
    results = [_r("a", "ok", [(0, 1)], [(0, 1)])]
    out = tmp_path / "failures.jsonl"
    n = write_failures(results, out, include_passes=True)
    assert n == 1
    [row] = _read_jsonl(out)
    assert row["false_positive_ranges"] == []
    assert row["false_negative_ranges"] == []


def test_multiple_articles_each_get_their_own_row(tmp_path: Path) -> None:
    results = [
        _r("clean", "hi", [], []),  # passes
        _r("fp", "abcde", [], [(0, 3)]),  # FP only
        _r("fn", "abcde", [(0, 3)], []),  # FN only
    ]
    out = tmp_path / "failures.jsonl"
    n = write_failures(results, out)
    assert n == 2  # "clean" excluded
    rows = _read_jsonl(out)
    by_id = {r["item_id"]: r for r in rows}
    assert "fp" in by_id and "fn" in by_id
    assert by_id["fp"]["false_positive_ranges"] == [[0, 3]]
    assert by_id["fn"]["false_negative_ranges"] == [[0, 3]]
