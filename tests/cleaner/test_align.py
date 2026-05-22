"""Tests for `data.align.tokenize_and_align` (TASK-009).

Uses a real DistilBERT tokenizer because the contract is specifically that
HF fast tokenizers' offset_mapping is in codepoints. Tests are skipped if
the tokenizer can't be downloaded (e.g. offline CI without an HF cache).
"""

from __future__ import annotations

import pytest

from alchimiste.cleaner.data.align import (
    LABEL_DROP,
    LABEL_IGNORE,
    LABEL_KEEP,
    TokenizedExample,
    reconstruct_ranges_from_token_labels,
    tokenize_and_align,
)
from alchimiste.cleaner.data.loader import LabeledArticle


@pytest.fixture(scope="module")
def tokenizer():
    transformers = pytest.importorskip("transformers")
    try:
        return transformers.AutoTokenizer.from_pretrained("distilbert-base-uncased")
    except Exception as e:  # network down, hub unavailable, etc.
        pytest.skip(f"could not load distilbert tokenizer: {e}")


def _article(text: str, ranges: list[tuple[int, int]] | None = None) -> LabeledArticle:
    return LabeledArticle(
        item_id="t",
        content_sha256="0" * 64,
        markdown_text=text,
        discard_ranges=tuple(ranges or []),
    )


def test_clean_article_has_no_drop_labels(tokenizer) -> None:
    art = _article("Hello world. Plain article.")
    ex = tokenize_and_align(art, tokenizer)
    assert isinstance(ex, TokenizedExample)
    assert LABEL_DROP not in ex.labels
    # Some tokens should still be present, with IGNORE for the specials.
    assert LABEL_IGNORE in ex.labels
    assert LABEL_KEEP in ex.labels


def test_drop_tokens_fall_inside_drop_ranges(tokenizer) -> None:
    """Property (a): every DROP-labeled token's interval is inside some range."""
    text = "Keep this. CLICK HERE TO SUBSCRIBE. Goodbye."
    s = text.index("CLICK")
    e = text.index("Goodbye")
    art = _article(text, [(s, e)])
    ex = tokenize_and_align(art, tokenizer)

    found_drop = False
    for label, (start, end) in zip(ex.labels, ex.codepoint_offset_mapping, strict=True):
        if label == LABEL_DROP:
            found_drop = True
            assert s <= start, f"drop token starts at {start} before range {s}"
            assert end <= e, f"drop token ends at {end} after range {e}"
    assert found_drop, "expected at least one DROP token in the boilerplate span"


def test_drop_runs_reconstruct_to_original_ranges(tokenizer) -> None:
    """Property (b): contiguous DROP runs ≈ the original discard_ranges (±1 cp at boundaries)."""
    text = "Keep this. CLICK HERE TO SUBSCRIBE. Goodbye."
    s = text.index("CLICK")
    e = text.index("Goodbye")
    art = _article(text, [(s, e)])
    ex = tokenize_and_align(art, tokenizer)

    reconstructed = reconstruct_ranges_from_token_labels(ex.labels, ex.codepoint_offset_mapping)
    assert len(reconstructed) == 1
    rs, re_ = reconstructed[0]
    # Within ±1 codepoint per boundary (spec acceptance criterion).
    assert abs(rs - s) <= 1
    assert abs(re_ - e) <= 1


def test_boundary_straddling_tokens_labeled_keep(tokenizer) -> None:
    """A token that overlaps a drop boundary but isn't fully inside stays KEEP."""
    # Pick a drop range that bisects a word so the tokenizer can't help but
    # straddle: drop the middle of "subscription".
    text = "Word subscription word."
    word_start = text.index("subscription")
    # Drop from middle of the word — guarantees the "subscription" token
    # straddles the boundary.
    drop_s = word_start + 3
    drop_e = word_start + 8
    art = _article(text, [(drop_s, drop_e)])
    ex = tokenize_and_align(art, tokenizer)

    # The tokenizer treats "subscription" as a single subword on DistilBERT;
    # its full offset span is wider than (drop_s, drop_e), so it must be KEEP.
    for label, (start, end) in zip(ex.labels, ex.codepoint_offset_mapping, strict=True):
        if start <= drop_s < end or start < drop_e <= end:
            # This token straddles a boundary.
            assert label == LABEL_KEEP, (
                f"straddling token at [{start}, {end}) should be KEEP, got {label}"
            )


def test_multi_byte_utf8_offsets_are_codepoints(tokenizer) -> None:
    """Verify offset_mapping is codepoint-indexed, not byte-indexed."""
    text = "Café. Drop me. Résumé."
    s = text.index("Drop")
    e = text.index("Résumé")
    art = _article(text, [(s, e)])
    ex = tokenize_and_align(art, tokenizer)

    # Reconstruct kept text from KEEP tokens — should not chop multibyte chars.
    reconstructed = reconstruct_ranges_from_token_labels(ex.labels, ex.codepoint_offset_mapping)
    assert len(reconstructed) == 1
    rs, re_ = reconstructed[0]
    assert abs(rs - s) <= 1
    assert abs(re_ - e) <= 1


def test_max_seq_len_truncates(tokenizer) -> None:
    long_text = ("the quick brown fox " * 200).strip()
    art = _article(long_text)
    ex = tokenize_and_align(art, tokenizer, max_seq_len=64)
    assert len(ex.input_ids) <= 64
    assert len(ex.labels) == len(ex.input_ids)
    assert len(ex.codepoint_offset_mapping) == len(ex.input_ids)


def test_empty_discard_ranges_yields_only_keep_and_ignore(tokenizer) -> None:
    art = _article("Just an article.")
    ex = tokenize_and_align(art, tokenizer)
    assert set(ex.labels).issubset({LABEL_KEEP, LABEL_IGNORE})


def test_reconstruct_multiple_disjoint_ranges(tokenizer) -> None:
    """Two non-adjacent drop ranges should round-trip to two reconstructed ranges."""
    text = "Keep one. CLICK HERE. Keep two. SUBSCRIBE NOW. Keep three."
    s1 = text.index("CLICK")
    e1 = text.index(". Keep two") + 1  # include the period
    s2 = text.index("SUBSCRIBE")
    e2 = text.index(". Keep three") + 1
    art = _article(text, [(s1, e1), (s2, e2)])
    ex = tokenize_and_align(art, tokenizer)

    reconstructed = reconstruct_ranges_from_token_labels(ex.labels, ex.codepoint_offset_mapping)
    assert len(reconstructed) == 2, f"expected 2 disjoint ranges, got {reconstructed}"


def test_reconstruct_skips_ignore_tokens_within_run() -> None:
    """LABEL_IGNORE mid-sequence must not split a DROP run (regression for align.py bug)."""
    # Synthetic: 3 DROP tokens with an IGNORE in the middle. The IGNORE has
    # offsets (0, 0) like a special token; without the fix, it would close
    # the run and produce two separate ranges instead of one.
    labels = [LABEL_IGNORE, LABEL_DROP, LABEL_IGNORE, LABEL_DROP, LABEL_DROP, LABEL_IGNORE]
    offsets = [(0, 0), (0, 4), (0, 0), (4, 8), (8, 12), (0, 0)]
    result = reconstruct_ranges_from_token_labels(labels, offsets)
    assert result == [(0, 12)], f"IGNORE inside a DROP run must not split it; got {result}"


def test_reconstruct_keep_closes_drop_run() -> None:
    """A KEEP token between two DROP tokens *does* close the run."""
    labels = [LABEL_DROP, LABEL_KEEP, LABEL_DROP]
    offsets = [(0, 4), (4, 8), (8, 12)]
    result = reconstruct_ranges_from_token_labels(labels, offsets)
    assert result == [(0, 4), (8, 12)]
