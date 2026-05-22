"""Tests for `inference.decode.decode_token_runs` (TASK-023)."""

from __future__ import annotations

import random
from itertools import pairwise

import pytest

from alchimiste.cleaner.inference.decode import decode_token_runs


def test_decode_basic_run() -> None:
    probs = [0.9, 0.9, 0.1, 0.9]
    offsets = [(0, 4), (4, 8), (8, 12), (12, 16)]
    ranges = decode_token_runs(probs, offsets, threshold=0.5)
    assert ranges == [(0, 8), (12, 16)]


def test_decode_skips_special_tokens() -> None:
    """Special tokens at (0,0) shouldn't break a run."""
    probs = [0.0, 0.9, 0.0, 0.9, 0.9, 0.0]
    offsets = [(0, 0), (0, 4), (0, 0), (4, 8), (8, 12), (0, 0)]
    ranges = decode_token_runs(probs, offsets, threshold=0.5)
    assert ranges == [(0, 12)]


def test_decode_threshold_filters() -> None:
    probs = [0.6, 0.4]
    offsets = [(0, 5), (5, 10)]
    assert decode_token_runs(probs, offsets, threshold=0.5) == [(0, 5)]
    assert decode_token_runs(probs, offsets, threshold=0.7) == []


def test_decode_min_run_len_filter() -> None:
    probs = [0.9, 0.1, 0.9, 0.9]
    offsets = [(0, 3), (3, 6), (6, 9), (9, 12)]
    # Run 1: (0,3) length=3; Run 2: (6,12) length=6.
    ranges = decode_token_runs(probs, offsets, threshold=0.5, min_run_len=4)
    assert ranges == [(6, 12)]


def test_decode_merges_adjacent_subword_runs() -> None:
    """Tokens that come out as separate but touching get merged after decoding."""
    probs = [0.9, 0.9]
    # Two tokens that touch at offset 5.
    offsets = [(0, 5), (5, 10)]
    ranges = decode_token_runs(probs, offsets, threshold=0.5)
    assert ranges == [(0, 10)]


def test_decode_empty_input() -> None:
    assert decode_token_runs([], [], threshold=0.5) == []


def test_decode_misaligned_inputs_raises() -> None:
    with pytest.raises(ValueError, match="must align"):
        decode_token_runs([0.5, 0.5], [(0, 1)], threshold=0.5)


def test_decode_property_sorted_non_overlapping(seed: int = 0) -> None:
    """Random fuzz: outputs are sorted, non-overlapping, in bounds."""
    rnd = random.Random(seed)
    for _ in range(10):
        n = rnd.randint(0, 30)
        # Construct contiguous offset map.
        offsets: list[tuple[int, int]] = []
        cursor = 0
        for _i in range(n):
            step = rnd.randint(1, 4)
            offsets.append((cursor, cursor + step))
            cursor += step
        probs = [rnd.random() for _ in range(n)]
        ranges = decode_token_runs(probs, offsets, threshold=0.5)
        # Sorted by start, non-overlapping.
        for a, b in pairwise(ranges):
            assert a[1] <= b[0], f"overlap or unsorted: {ranges}"
        for s, e in ranges:
            assert 0 <= s < e <= cursor, f"out of bounds: {(s, e)} text_len={cursor}"
