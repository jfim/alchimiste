"""Per-token drop probabilities -> codepoint ranges (TASK-023, REQ-012, design.md § 4.3).

The decoder is intentionally pure and tiny — it's used both by the
mlflow pyfunc at inference time and by threshold selection on the
validation set, so the two share one algorithm.

Algorithm:
  1. Threshold per-token drop probabilities.
  2. Group consecutive drop-tokens into runs (special tokens whose
     offset is (0, 0) are skipped — they don't terminate runs).
  3. For each run, emit `[min(start), max(end))` over the codepoint
     offsets.
  4. Drop runs shorter than `min_run_len` codepoints.
  5. Merge any adjacent/overlapping ranges produced by sub-word
     tokenizers.
"""

from __future__ import annotations

from collections.abc import Sequence

Range = tuple[int, int]


def decode_token_runs(
    probs: Sequence[float],
    codepoint_offset_mapping: Sequence[Range],
    *,
    threshold: float,
    min_run_len: int = 0,
) -> list[Range]:
    """Convert per-token drop probabilities into sorted, non-overlapping
    codepoint ranges. See module docstring for the algorithm."""
    if len(probs) != len(codepoint_offset_mapping):
        raise ValueError(
            f"probs (len={len(probs)}) and offset mapping "
            f"(len={len(codepoint_offset_mapping)}) must align"
        )

    ranges: list[Range] = []
    in_run = False
    cur_start = 0
    cur_end = 0
    for p, (start, end) in zip(probs, codepoint_offset_mapping, strict=True):
        is_special = start == 0 and end == 0
        if is_special:
            # Don't terminate an active run on specials — see TASK-009.
            continue
        if p >= threshold:
            if in_run:
                cur_end = max(cur_end, end)
            else:
                in_run = True
                cur_start = start
                cur_end = end
        else:
            if in_run:
                ranges.append((cur_start, cur_end))
                in_run = False
    if in_run:
        ranges.append((cur_start, cur_end))

    if min_run_len > 0:
        ranges = [r for r in ranges if (r[1] - r[0]) >= min_run_len]

    return _merge_adjacent(ranges)


def _merge_adjacent(ranges: list[Range]) -> list[Range]:
    """Merge ranges that touch or overlap. Input must be sorted by start."""
    if not ranges:
        return ranges
    merged: list[Range] = [ranges[0]]
    for start, end in ranges[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end:  # touching (last_end == start) or overlapping
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged
