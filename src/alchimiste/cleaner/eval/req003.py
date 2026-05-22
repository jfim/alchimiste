"""Clean-only and excessive-ranges sub-evaluations (TASK-021, REQ-003).

Two derived numbers reported alongside the primary IoU metrics so the
labeler/trainer can spot regressions without staring at the full
metric table:

  * `clean_articles_kept_clean_fraction` — among articles where the
    ground truth has *no* drop ranges, what fraction does the model
    also predict as empty? High is good (REQ-003 acceptance gate
    target ≥ 0.95).
  * `excessive_ranges_fraction` — fraction of articles where the model
    predicts more than `excessive_ranges_threshold` drop ranges. This
    is a smell flag for human review, not a hard fail (REQ-003).
"""

from __future__ import annotations

from collections.abc import Sequence

Range = tuple[int, int]


def clean_articles_kept_clean_fraction(
    per_article_true: Sequence[Sequence[Range]],
    per_article_pred: Sequence[Sequence[Range]],
) -> float:
    """REQ-003 (a): of the articles with empty ground-truth ranges, what
    fraction does the model also predict as empty?"""
    if len(per_article_true) != len(per_article_pred):
        raise ValueError("per-article true and pred sequences must align")
    n_clean = 0
    n_kept = 0
    for t, p in zip(per_article_true, per_article_pred, strict=True):
        if len(t) == 0:
            n_clean += 1
            if len(p) == 0:
                n_kept += 1
    if n_clean == 0:
        return 0.0
    return n_kept / n_clean


def excessive_ranges_fraction(
    per_article_pred: Sequence[Sequence[Range]],
    *,
    excessive_ranges_threshold: int = 6,
) -> float:
    """REQ-003 (b): fraction of articles where the model predicted more
    than `excessive_ranges_threshold` drop ranges. Used as a smell flag
    for human review, not a hard failure."""
    if not per_article_pred:
        return 0.0
    n_excessive = sum(1 for p in per_article_pred if len(p) > excessive_ranges_threshold)
    return n_excessive / len(per_article_pred)
