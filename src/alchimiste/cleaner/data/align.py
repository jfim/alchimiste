"""Codepoint↔token alignment (REQ-004, design.md § 2.5).

Given a `LabeledArticle` (NFC text + codepoint-offset `discard_ranges`) and a
HuggingFace fast tokenizer, build a `TokenizedExample` whose per-token labels
agree with the article's drop ranges.

Labeling rule (design.md § 2.5 + NFR-002 precision bias):
  * A token is labeled `drop` (1) iff its codepoint interval is **entirely**
    inside some `discard_range`.
  * Boundary-straddling tokens are labeled `keep` (0) — conservative bias
    against dropping real article text.
  * Special tokens (CLS, SEP, PAD) get label -100 so they're ignored by the
    standard cross-entropy loss.

The HF fast tokenizer's `offset_mapping` returns `(start_char, end_char)`
where the units are codepoints when the tokenizer is fed a Python `str` —
exactly the unit our labels use.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # avoid a hard transformers import at module load
    from transformers import PreTrainedTokenizerFast

from alchimiste.cleaner.data.loader import LabeledArticle

# Label conventions — keep / drop / ignore.
LABEL_KEEP = 0
LABEL_DROP = 1
LABEL_IGNORE = -100  # PyTorch CE loss `ignore_index` default


@dataclass(frozen=True)
class TokenizedExample:
    """One labeled article in the model's tokenization (design.md § 2.5)."""

    item_id: str
    input_ids: tuple[int, ...]
    codepoint_offset_mapping: tuple[tuple[int, int], ...]
    labels: tuple[int, ...]


def tokenize_and_align(
    article: LabeledArticle,
    tokenizer: PreTrainedTokenizerFast,
    *,
    max_seq_len: int = 512,
) -> TokenizedExample:
    """Tokenize `article.markdown_text` and project `discard_ranges` to labels.

    Parameters
    ----------
    article
        Source article (NFC text + codepoint-offset discard ranges).
    tokenizer
        Any HF *fast* tokenizer supporting `return_offsets_mapping=True`.
    max_seq_len
        Truncate to this many tokens (design.md § 6 / config: `data.max_seq_len`).
        Articles exceeding this are silently truncated; an upstream caller is
        expected to track truncations for Open Question Q2.

    Returns
    -------
    TokenizedExample
        Tuple-typed (frozen) container ready for batching.
    """
    enc = tokenizer(
        article.markdown_text,
        return_offsets_mapping=True,
        truncation=True,
        max_length=max_seq_len,
        return_tensors=None,
        add_special_tokens=True,
    )
    input_ids = list(enc["input_ids"])
    offsets = list(enc["offset_mapping"])
    seq_ids = enc.sequence_ids()

    labels = _project_labels(
        offsets=offsets,
        seq_ids=seq_ids,
        discard_ranges=article.discard_ranges,
    )

    return TokenizedExample(
        item_id=article.item_id,
        input_ids=tuple(input_ids),
        codepoint_offset_mapping=tuple((int(s), int(e)) for s, e in offsets),
        labels=tuple(labels),
    )


def _project_labels(
    *,
    offsets: list[tuple[int, int]],
    seq_ids: list[int | None],
    discard_ranges: tuple[tuple[int, int], ...],
) -> list[int]:
    """Project codepoint-range labels onto the token stream.

    A token is `drop` iff its full `[start, end)` interval is contained in
    some `discard_range`. Special tokens (`seq_ids[i] is None`) become
    `LABEL_IGNORE`.
    """
    out: list[int] = []
    for i, (start, end) in enumerate(offsets):
        if seq_ids[i] is None:
            out.append(LABEL_IGNORE)
            continue
        if _fully_inside_any(start, end, discard_ranges):
            out.append(LABEL_DROP)
        else:
            out.append(LABEL_KEEP)
    return out


def _fully_inside_any(
    start: int,
    end: int,
    ranges: tuple[tuple[int, int], ...],
) -> bool:
    """True iff `[start, end)` is fully contained in some range in `ranges`.

    Ranges are sorted non-overlapping (loader invariant), so we could short-
    circuit with bisect — but the per-article range count is tiny in practice,
    so a linear scan is fine and keeps this dependency-free.
    """
    return any(r_start <= start and end <= r_end for r_start, r_end in ranges)


def reconstruct_ranges_from_token_labels(
    labels: tuple[int, ...] | list[int],
    offsets: tuple[tuple[int, int], ...] | list[tuple[int, int]],
) -> list[tuple[int, int]]:
    """Inverse of `_project_labels`: contiguous drop-token runs → codepoint ranges.

    Used by tests; design.md § 4.3's decoder shares the same algorithm.

    `LABEL_IGNORE` tokens (specials such as CLS/SEP/PAD) are *skipped* —
    they do not close an active DROP run. This matters when specials can
    appear mid-sequence (e.g. SEP between two segments); for the common
    BERT-style at-the-ends layout the behavior is indistinguishable.
    `LABEL_KEEP` tokens *do* close an active DROP run.
    """
    ranges: list[tuple[int, int]] = []
    in_run = False
    cur_start = 0
    cur_end = 0
    for label, (start, end) in zip(labels, offsets, strict=True):
        if label == LABEL_IGNORE:
            continue
        if label == LABEL_DROP:
            if in_run:
                cur_end = max(cur_end, end)
            else:
                in_run = True
                cur_start = start
                cur_end = end
        else:  # LABEL_KEEP — close any open DROP run
            if in_run:
                ranges.append((cur_start, cur_end))
                in_run = False
    if in_run:
        ranges.append((cur_start, cur_end))
    return ranges
