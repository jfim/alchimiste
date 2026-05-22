"""Tests for the truth-clipping helpers in `training.finalize`.

When `max_seq_len` truncates an article, drop ranges past the truncation
point are unreachable for the model. Scoring should clip the truth to
the visible window so the model isn't penalized for tokens it never saw.
"""

from __future__ import annotations

from alchimiste.cleaner.data.align import TokenizedExample
from alchimiste.cleaner.training.finalize import (
    _clip_ranges_to_window,
    _example_window_end,
)


def _example(offsets: list[tuple[int, int]]) -> TokenizedExample:
    return TokenizedExample(
        item_id="x",
        input_ids=tuple(0 for _ in offsets),
        codepoint_offset_mapping=tuple(offsets),
        labels=tuple(0 for _ in offsets),
    )


def test_window_end_uses_max_offset() -> None:
    # SEP tokens get (0, 0); the visible end is the largest end offset.
    ex = _example([(0, 0), (0, 5), (5, 12), (12, 20), (0, 0)])
    assert _example_window_end(ex) == 20


def test_window_end_empty_is_zero() -> None:
    assert _example_window_end(_example([])) == 0


def test_clip_preserves_ranges_fully_inside() -> None:
    assert _clip_ranges_to_window([(0, 5), (10, 15)], window_end=20) == [(0, 5), (10, 15)]


def test_clip_drops_ranges_fully_past_window() -> None:
    assert _clip_ranges_to_window([(0, 5), (100, 200)], window_end=50) == [(0, 5)]


def test_clip_truncates_straddling_range() -> None:
    # Range [40, 80) straddles a window ending at 50 → clipped to [40, 50).
    assert _clip_ranges_to_window([(40, 80)], window_end=50) == [(40, 50)]


def test_clip_drops_range_starting_at_window_end() -> None:
    # Half-open: [50, 60) with window_end=50 has zero overlap.
    assert _clip_ranges_to_window([(50, 60)], window_end=50) == []


def test_clip_empty_ranges() -> None:
    assert _clip_ranges_to_window([], window_end=100) == []
