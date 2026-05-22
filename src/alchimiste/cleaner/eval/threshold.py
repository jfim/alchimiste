"""Validation-set threshold selection (TASK-020, REQ-007, NFR-002, design.md § 5.3).

Sweeps a configurable τ grid on the validation set, decodes drop ranges
at each τ, and picks the lowest τ that meets the precision floor at
IoU=0.5. Falls back to the τ with maximum precision when none meets
the floor (and reports it via a tag on the mlflow run).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from alchimiste.cleaner.eval.iou import aggregate_iou_metrics
from alchimiste.cleaner.inference.decode import decode_token_runs

Range = tuple[int, int]


@dataclass(frozen=True)
class ThresholdSelection:
    """Result of a sweep — the chosen τ and how it was chosen."""

    threshold: float
    iou_metric: float
    precision_at_chosen: float
    recall_at_chosen: float
    f1_at_chosen: float
    fell_back_to_max_precision: bool
    sweep: tuple[tuple[float, float, float, float], ...]
    """One row per τ swept: (threshold, precision, recall, f1)."""


def sweep_thresholds(
    per_example_probs: Sequence[Sequence[float]],
    per_example_offsets: Sequence[Sequence[Range]],
    per_example_true_ranges: Sequence[Sequence[Range]],
    *,
    sweep_min: float = 0.30,
    sweep_max: float = 0.90,
    sweep_step: float = 0.05,
    iou_metric: float = 0.5,
    precision_floor: float = 0.85,
) -> ThresholdSelection:
    """Pick the lowest τ that meets `precision_floor` at the given IoU level.

    All sequences must have the same length (one entry per validation
    example). Returns the chosen threshold plus the full sweep so the
    training loop can log it.
    """
    if not (len(per_example_probs) == len(per_example_offsets) == len(per_example_true_ranges)):
        raise ValueError("per-example sequences must all be the same length")

    thresholds = _arange(sweep_min, sweep_max, sweep_step)
    rows: list[tuple[float, float, float, float]] = []
    for tau in thresholds:
        pred_per_article: list[list[Range]] = [
            decode_token_runs(probs, offsets, threshold=tau)
            for probs, offsets in zip(per_example_probs, per_example_offsets, strict=True)
        ]
        agg = aggregate_iou_metrics(
            per_example_true_ranges,
            pred_per_article,
            thresholds=(iou_metric,),
        )[iou_metric]
        rows.append((tau, agg.precision, agg.recall, agg.f1))

    # Pick the lowest τ meeting the precision floor.
    meeting = [r for r in rows if r[1] >= precision_floor]
    if meeting:
        chosen = min(meeting, key=lambda r: r[0])
        fell_back = False
    else:
        # No τ meets the floor — take the one with max precision (ties
        # broken by recall, then by lowest threshold for determinism).
        chosen = max(rows, key=lambda r: (r[1], r[2], -r[0]))
        fell_back = True

    return ThresholdSelection(
        threshold=chosen[0],
        iou_metric=iou_metric,
        precision_at_chosen=chosen[1],
        recall_at_chosen=chosen[2],
        f1_at_chosen=chosen[3],
        fell_back_to_max_precision=fell_back,
        sweep=tuple(rows),
    )


def _arange(start: float, stop: float, step: float) -> list[float]:
    """Inclusive of `stop` to within a tiny epsilon (defending against
    floating-point drift in the loop)."""
    out: list[float] = []
    cur = start
    eps = step * 1e-6
    while cur <= stop + eps:
        # Round to 6 decimal places to keep the threshold keys readable
        # in the swept rows and the mlflow run.
        out.append(round(cur, 6))
        cur += step
    return out
