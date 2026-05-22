"""Per-article failure dump in an alambic-re-ingestible JSONL form (TASK-019, REQ-010).

Schema (one JSON object per line, see design.md § 7.4):

    {
      "item_id": "...",
      "content_sha256": "...",
      "true_drop_ranges": [[s, e], ...],
      "pred_drop_ranges": [[s, e], ...],
      "false_positive_ranges": [[s, e], ...],
      "false_negative_ranges": [[s, e], ...],
      "kept_diff": "unified-diff-style rendering"
    }

`false_positive_ranges` / `false_negative_ranges` are computed at the
IoU=0.5 threshold (REQ-009 primary metric). The diff is over the
"kept" text — what's left after each side's drop ranges are applied.
"""

from __future__ import annotations

import difflib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from alchimiste.cleaner.data.loader import apply_discard_ranges
from alchimiste.cleaner.eval.iou import iou

Range = tuple[int, int]


@dataclass(frozen=True)
class ArticleResult:
    """Per-article slice of the test set needed to write a failure row."""

    item_id: str
    content_sha256: str
    text: str
    true_drop_ranges: tuple[Range, ...]
    pred_drop_ranges: tuple[Range, ...]


def write_failures(
    results: Sequence[ArticleResult],
    dst: Path,
    *,
    iou_threshold: float = 0.5,
    include_passes: bool = False,
) -> int:
    """Write one JSONL row per failing article to `dst`. Returns count written.

    A "failure" is any article where the predicted ranges don't perfectly
    match the truth at `iou_threshold` (i.e. there's at least one FP or
    FN). Set `include_passes=True` to write every article unconditionally
    — handy for offline analysis.
    """
    n = 0
    with dst.open("w", encoding="utf-8") as f:
        for r in results:
            fps, fns = _classify(r.pred_drop_ranges, r.true_drop_ranges, iou_threshold)
            if not include_passes and not fps and not fns:
                continue
            row = {
                "item_id": r.item_id,
                "content_sha256": r.content_sha256,
                "true_drop_ranges": [list(p) for p in r.true_drop_ranges],
                "pred_drop_ranges": [list(p) for p in r.pred_drop_ranges],
                "false_positive_ranges": [list(p) for p in fps],
                "false_negative_ranges": [list(p) for p in fns],
                "kept_diff": _kept_diff(
                    text=r.text,
                    true_ranges=r.true_drop_ranges,
                    pred_ranges=r.pred_drop_ranges,
                ),
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def _classify(
    pred_ranges: Sequence[Range],
    true_ranges: Sequence[Range],
    threshold: float,
) -> tuple[list[Range], list[Range]]:
    """Greedy 1-to-1 matching at IoU=`threshold` (mirrors eval/iou.py).
    Returns (false_positives, false_negatives)."""
    matched_true: set[int] = set()
    fps: list[Range] = []
    for p in pred_ranges:
        best_iou = 0.0
        best_idx: int | None = None
        for j, t in enumerate(true_ranges):
            if j in matched_true:
                continue
            v = iou(p, t)
            if v > best_iou:
                best_iou = v
                best_idx = j
        if best_idx is not None and best_iou >= threshold:
            matched_true.add(best_idx)
        else:
            fps.append(tuple(p))
    fns = [tuple(t) for j, t in enumerate(true_ranges) if j not in matched_true]
    return fps, fns


def _kept_diff(*, text: str, true_ranges: Sequence[Range], pred_ranges: Sequence[Range]) -> str:
    true_kept = apply_discard_ranges(text, true_ranges).splitlines(keepends=True)
    pred_kept = apply_discard_ranges(text, pred_ranges).splitlines(keepends=True)
    return "".join(
        difflib.unified_diff(
            true_kept,
            pred_kept,
            fromfile="true_kept",
            tofile="pred_kept",
            n=2,
        )
    )
