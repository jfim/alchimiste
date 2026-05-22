"""IoU-based metrics over codepoint ranges (REQ-009, design.md § 7.1).

The primary evaluation metric for the cleaner is IoU over predicted vs.
ground-truth drop ranges. For each predicted range we find the
ground-truth range with the maximum IoU; at each threshold τ we count
predictions as true positives iff that best-IoU ≥ τ.

We report at τ ∈ {0.5, 0.9, 1.0}:
  * 0.5 — lenient: predictions roughly aligned with truth count.
  * 0.9 — strict: predictions must overlap nearly perfectly.
  * 1.0 — exact-match: equal half-open intervals.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

Range = tuple[int, int]


@dataclass(frozen=True)
class IoUMetrics:
    """precision/recall/F1 of the drop class at one IoU threshold."""

    threshold: float
    tp: int
    fp: int
    fn: int
    precision: float
    recall: float
    f1: float


def iou(a: Range, b: Range) -> float:
    """Half-open-interval IoU. Returns 0.0 for disjoint ranges."""
    inter = max(0, min(a[1], b[1]) - max(a[0], b[0]))
    if inter == 0:
        return 0.0
    union = (a[1] - a[0]) + (b[1] - b[0]) - inter
    return inter / union if union > 0 else 0.0


def iou_metrics(
    true_ranges: Sequence[Range],
    pred_ranges: Sequence[Range],
    *,
    thresholds: Sequence[float] = (0.5, 0.9, 1.0),
) -> dict[float, IoUMetrics]:
    """Compute IoU-based precision/recall/F1 at each threshold.

    Matching is greedy and 1-to-1: each predicted range is paired with
    its best-IoU ground-truth range; that ground-truth range is then
    consumed (can't pair with another pred) so we don't inflate recall
    by double-counting.
    """
    out: dict[float, IoUMetrics] = {}
    for tau in thresholds:
        out[tau] = _metrics_at_threshold(true_ranges, pred_ranges, tau)
    return out


def _metrics_at_threshold(
    true_ranges: Sequence[Range],
    pred_ranges: Sequence[Range],
    tau: float,
) -> IoUMetrics:
    matched_truth: set[int] = set()
    tp = 0
    for pred in pred_ranges:
        best_iou = 0.0
        best_idx: int | None = None
        for j, truth in enumerate(true_ranges):
            if j in matched_truth:
                continue
            i = iou(pred, truth)
            if i > best_iou:
                best_iou = i
                best_idx = j
        if best_idx is not None and best_iou >= tau:
            tp += 1
            matched_truth.add(best_idx)
    fp = len(pred_ranges) - tp
    fn = len(true_ranges) - tp
    precision = tp / max(tp + fp, 1) if (tp + fp) else 0.0
    recall = tp / max(tp + fn, 1) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return IoUMetrics(threshold=tau, tp=tp, fp=fp, fn=fn, precision=precision, recall=recall, f1=f1)


def aggregate_iou_metrics(
    per_article_true: Sequence[Sequence[Range]],
    per_article_pred: Sequence[Sequence[Range]],
    *,
    thresholds: Sequence[float] = (0.5, 0.9, 1.0),
) -> dict[float, IoUMetrics]:
    """Micro-average IoU metrics across all articles in a test split.

    Computes per-article TPs/FPs/FNs at each threshold and sums them,
    then derives precision/recall/F1 from the totals. Micro-averaging
    weights articles by their range counts, which is what we want — a
    boilerplate-heavy article shouldn't be dominated by a clean one
    that has no ranges to score.
    """
    if len(per_article_true) != len(per_article_pred):
        raise ValueError(
            f"per-article lengths differ: {len(per_article_true)} true vs "
            f"{len(per_article_pred)} pred"
        )
    totals: dict[float, list[int]] = {tau: [0, 0, 0] for tau in thresholds}  # tp, fp, fn
    for t_ranges, p_ranges in zip(per_article_true, per_article_pred, strict=True):
        per_thr = iou_metrics(t_ranges, p_ranges, thresholds=thresholds)
        for tau, m in per_thr.items():
            totals[tau][0] += m.tp
            totals[tau][1] += m.fp
            totals[tau][2] += m.fn

    out: dict[float, IoUMetrics] = {}
    for tau, (tp, fp, fn) in totals.items():
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        out[tau] = IoUMetrics(
            threshold=tau, tp=tp, fp=fp, fn=fn, precision=precision, recall=recall, f1=f1
        )
    return out
