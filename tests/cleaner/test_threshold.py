"""Tests for `eval.threshold.sweep_thresholds` (TASK-020)."""

from __future__ import annotations

from alchimiste.cleaner.eval.threshold import sweep_thresholds


def _example(prob_values: list[float], n_tokens: int | None = None):
    """Build a single fake example: probs + contiguous offsets."""
    n = n_tokens or len(prob_values)
    offsets = [(i, i + 1) for i in range(n)]
    return prob_values, offsets


def test_picks_lowest_threshold_meeting_precision_floor() -> None:
    """At low τ the model emits lots of FPs (low precision); at high τ
    it emits fewer (higher precision). The sweep should pick the lowest
    τ that crosses the floor."""
    probs1, offs1 = _example([0.8, 0.8, 0.8, 0.3])
    true1 = [(0, 3)]
    sel = sweep_thresholds(
        per_example_probs=[probs1],
        per_example_offsets=[offs1],
        per_example_true_ranges=[true1],
        sweep_min=0.2,
        sweep_max=0.9,
        sweep_step=0.1,
        iou_metric=0.5,
        precision_floor=0.5,
    )
    assert not sel.fell_back_to_max_precision
    assert sel.precision_at_chosen >= 0.5


def test_falls_back_when_no_threshold_meets_floor() -> None:
    """Force precision below the floor at every τ by producing many FPs."""
    # All-low probs with high precision floor that nothing can meet.
    probs, offs = _example([0.1] * 10)
    true_ = [(0, 10)]
    sel = sweep_thresholds(
        per_example_probs=[probs],
        per_example_offsets=[offs],
        per_example_true_ranges=[true_],
        sweep_min=0.5,
        sweep_max=0.9,
        sweep_step=0.1,
        iou_metric=0.5,
        precision_floor=0.99,
    )
    # No predictions at all -> precision=0 everywhere; must fall back.
    assert sel.fell_back_to_max_precision


def test_sweep_is_deterministic() -> None:
    probs, offs = _example([0.7, 0.7, 0.2])
    true_ = [(0, 2)]
    a = sweep_thresholds(
        per_example_probs=[probs],
        per_example_offsets=[offs],
        per_example_true_ranges=[true_],
        sweep_step=0.1,
    )
    b = sweep_thresholds(
        per_example_probs=[probs],
        per_example_offsets=[offs],
        per_example_true_ranges=[true_],
        sweep_step=0.1,
    )
    assert a == b


def test_sweep_row_count_matches_grid() -> None:
    probs, offs = _example([0.5])
    sel = sweep_thresholds(
        per_example_probs=[probs],
        per_example_offsets=[offs],
        per_example_true_ranges=[[]],
        sweep_min=0.3,
        sweep_max=0.9,
        sweep_step=0.1,
    )
    # 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9 -> 7 rows
    assert len(sel.sweep) == 7
