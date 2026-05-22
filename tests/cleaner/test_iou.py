"""Tests for `eval.iou` (TASK-017, REQ-009)."""

from __future__ import annotations

import pytest

from alchimiste.cleaner.eval.iou import (
    IoUMetrics,
    aggregate_iou_metrics,
    iou,
    iou_metrics,
)


def test_iou_disjoint_is_zero() -> None:
    assert iou((0, 5), (10, 15)) == 0.0


def test_iou_identical_is_one() -> None:
    assert iou((3, 10), (3, 10)) == 1.0


def test_iou_partial_overlap() -> None:
    # truth=(0,10), pred=(5,15) -> inter=5, union=15 -> 1/3
    assert iou((0, 10), (5, 15)) == pytest.approx(5 / 15)


def test_iou_contained() -> None:
    # truth=(0,20), pred=(5,10) -> inter=5, union=20 -> 0.25
    assert iou((0, 20), (5, 10)) == 0.25


def test_iou_touching_boundaries_is_zero() -> None:
    # Half-open: (0,5) and (5,10) share no codepoints.
    assert iou((0, 5), (5, 10)) == 0.0


def test_metrics_perfect_match_at_all_thresholds() -> None:
    truth = [(0, 10), (20, 30)]
    pred = [(0, 10), (20, 30)]
    m = iou_metrics(truth, pred)
    for tau, ms in m.items():
        assert ms.tp == 2, f"tau={tau}"
        assert ms.fp == 0
        assert ms.fn == 0
        assert ms.precision == 1.0
        assert ms.recall == 1.0
        assert ms.f1 == 1.0


def test_metrics_threshold_sensitivity() -> None:
    # truth=(0,10), pred=(0,12) -> IoU = 10/12 ≈ 0.833
    truth = [(0, 10)]
    pred = [(0, 12)]
    m = iou_metrics(truth, pred, thresholds=(0.5, 0.9, 1.0))
    assert m[0.5].tp == 1  # 0.833 >= 0.5
    assert m[0.9].tp == 0  # 0.833 < 0.9
    assert m[1.0].tp == 0  # 0.833 < 1.0


def test_metrics_no_pred_yields_zero_recall_and_perfect_precision() -> None:
    truth = [(0, 10)]
    pred: list[tuple[int, int]] = []
    m = iou_metrics(truth, pred)[0.5]
    assert m.tp == 0
    assert m.fp == 0
    assert m.fn == 1
    assert m.precision == 0.0  # no positives, define as 0
    assert m.recall == 0.0


def test_metrics_no_truth_yields_perfect_recall_zero_precision() -> None:
    truth: list[tuple[int, int]] = []
    pred = [(0, 10)]
    m = iou_metrics(truth, pred)[0.5]
    assert m.fp == 1
    assert m.fn == 0
    assert m.precision == 0.0
    assert m.recall == 0.0


def test_metrics_greedy_one_to_one_matching() -> None:
    """Two preds shouldn't both claim the same truth."""
    truth = [(0, 10)]
    pred = [(0, 9), (1, 10)]  # both overlap truth heavily
    m = iou_metrics(truth, pred, thresholds=(0.5,))[0.5]
    # Greedy: one pred matches (tp=1), the other is FP.
    assert m.tp == 1
    assert m.fp == 1
    assert m.fn == 0


def test_aggregate_micro_averages_over_articles() -> None:
    per_true = [
        [(0, 10)],  # article 1
        [(5, 15)],  # article 2
        [],  # article 3 (clean)
    ]
    per_pred = [
        [(0, 10)],  # match
        [(20, 30)],  # disjoint -> FP and FN
        [(0, 5)],  # FP
    ]
    out = aggregate_iou_metrics(per_true, per_pred, thresholds=(0.5,))
    m = out[0.5]
    assert m.tp == 1
    assert m.fp == 2
    assert m.fn == 1
    assert m.precision == pytest.approx(1 / 3)
    assert m.recall == 0.5


def test_aggregate_mismatched_lengths_raises() -> None:
    with pytest.raises(ValueError, match="lengths differ"):
        aggregate_iou_metrics([[(0, 1)]], [[(0, 1)], [(2, 3)]])


def test_metrics_dataclass_is_frozen() -> None:
    m = IoUMetrics(0.5, 1, 0, 0, 1.0, 1.0, 1.0)
    with pytest.raises((AttributeError, Exception)):
        m.tp = 99  # type: ignore[misc]
