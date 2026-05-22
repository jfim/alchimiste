"""Tests for REQ-003 sub-evaluations (TASK-021)."""

from __future__ import annotations

from alchimiste.cleaner.eval.req003 import (
    clean_articles_kept_clean_fraction,
    excessive_ranges_fraction,
)


def test_clean_kept_clean_all_clean_all_kept() -> None:
    true_ = [[], [], []]
    pred = [[], [], []]
    assert clean_articles_kept_clean_fraction(true_, pred) == 1.0


def test_clean_kept_clean_some_polluted() -> None:
    true_ = [[], [], [(0, 1)]]
    pred = [[(0, 1)], [], []]  # first article "polluted", third article "missed"
    # 2 clean articles, 1 of them kept clean -> 0.5
    assert clean_articles_kept_clean_fraction(true_, pred) == 0.5


def test_clean_kept_clean_no_clean_articles() -> None:
    true_ = [[(0, 1)], [(2, 3)]]
    pred = [[], []]
    # No clean articles at all -> return 0.0 (denominator is 0).
    assert clean_articles_kept_clean_fraction(true_, pred) == 0.0


def test_excessive_ranges_zero_when_below_threshold() -> None:
    pred = [[(0, 1)], [(0, 1), (2, 3)], []]
    # All have <= 6 ranges -> 0
    assert excessive_ranges_fraction(pred, excessive_ranges_threshold=6) == 0.0


def test_excessive_ranges_some_over_threshold() -> None:
    # Article with 8 ranges (> 6) and one with 2.
    pred = [
        [(i * 2, i * 2 + 1) for i in range(8)],
        [(0, 1), (2, 3)],
    ]
    assert excessive_ranges_fraction(pred, excessive_ranges_threshold=6) == 0.5


def test_excessive_ranges_empty_corpus() -> None:
    assert excessive_ranges_fraction([], excessive_ranges_threshold=6) == 0.0
