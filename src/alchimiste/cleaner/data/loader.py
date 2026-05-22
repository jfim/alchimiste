"""Load the `cleaning` dataset from an oxen working tree (REQ-001/002/013).

Reads `<oxen-dir>/<stage>/rows.parquet` via polars, joins each row to its
blob at `<oxen-dir>/<stage>/blobs/<content_sha256>`, decodes the blob as
UTF-8, applies NFC handling per `require_nfc`, validates the
`discard_ranges` invariants, and yields one `LabeledArticle` per row.

The `range_units` parameter is the transitional knob for Open Question Q3
(design.md § 2.3 / requirements.md REQ-002): alambic currently exports byte
offsets; the target contract is codepoint offsets over NFC text. While the
contract is settling, we accept either form on the wire and convert to
codepoint offsets internally.
"""

from __future__ import annotations

import warnings
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import polars as pl

from alchimiste.cleaner.data.normalize import NotNFCError, assert_nfc, defensive_nfc

RangeUnits = Literal["byte", "codepoint"]


@dataclass(frozen=True)
class LabeledArticle:
    """One labeled cleaning example, post-NFC, with codepoint-offset ranges.

    Fields follow design.md § 2.4. `discard_ranges` is always codepoint
    offsets into `markdown_text` (the loader converts from byte offsets
    when needed).
    """

    item_id: str
    content_sha256: str
    markdown_text: str
    discard_ranges: tuple[tuple[int, int], ...]


def load_oxen_tree(
    oxen_dir: Path | str,
    stage: str = "cleaning",
    *,
    require_nfc: bool = True,
    range_units: RangeUnits = "byte",
) -> list[LabeledArticle]:
    """Load every row in `<oxen_dir>/<stage>/rows.parquet` as a `LabeledArticle`.

    Parameters
    ----------
    oxen_dir
        Path to the oxen working tree (the dataset-puller's destination).
    stage
        Which stage to load. v1 only consumes "cleaning".
    require_nfc
        When True (default, REQ-013), each blob must already be NFC and
        the loader raises on any that isn't. When False, the loader
        defensively normalizes and emits a warning per offending blob
        (the per-row labels may then be slightly misaligned).
    range_units
        On-the-wire units of the exported `discard_ranges` column.
        "byte" matches today's alambic export; the loader converts to
        codepoint offsets via UTF-8 indexing on the NFC text. "codepoint"
        is the target contract once alambic adopts it.

    Raises
    ------
    FileNotFoundError
        If the parquet file or any referenced blob is missing.
    ValueError
        For schema mismatches, out-of-bounds / overlapping ranges, or
        byte offsets that don't fall on UTF-8 codepoint boundaries.
    NotNFCError
        When `require_nfc=True` and a blob fails the NFC check.
    """
    oxen_dir = Path(oxen_dir)
    stage_dir = oxen_dir / stage
    parquet_path = stage_dir / "rows.parquet"
    blobs_dir = stage_dir / "blobs"

    if not parquet_path.exists():
        raise FileNotFoundError(f"missing rows.parquet at {parquet_path}")
    if not blobs_dir.exists():
        raise FileNotFoundError(f"missing blobs directory at {blobs_dir}")

    df = pl.read_parquet(parquet_path)
    _validate_schema(df)

    articles: list[LabeledArticle] = []
    for row in df.iter_rows(named=True):
        articles.append(
            _load_row(
                row,
                blobs_dir=blobs_dir,
                require_nfc=require_nfc,
                range_units=range_units,
            )
        )
    return articles


# --------------------------------------------------------------------------- #
# Internals                                                                   #
# --------------------------------------------------------------------------- #

_REQUIRED_COLUMNS = ("item_id", "content_sha256", "discard_ranges")


def _validate_schema(df: pl.DataFrame) -> None:
    missing = [c for c in _REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"rows.parquet missing required columns: {missing}")


def _load_row(
    row: dict,
    *,
    blobs_dir: Path,
    require_nfc: bool,
    range_units: RangeUnits,
) -> LabeledArticle:
    item_id = row["item_id"]
    sha = row["content_sha256"]
    raw_ranges = row["discard_ranges"] or []

    blob_path = blobs_dir / sha
    if not blob_path.exists():
        raise FileNotFoundError(f"missing blob {sha} for item_id={item_id}")

    raw_bytes = blob_path.read_bytes()
    text = raw_bytes.decode("utf-8")

    if require_nfc:
        try:
            assert_nfc(text)
        except NotNFCError as e:
            raise NotNFCError(f"blob {sha} (item_id={item_id}) is not in NFC") from e
    else:
        new_text = defensive_nfc(text)
        if new_text != text:
            warnings.warn(
                f"blob {sha} (item_id={item_id}) was not NFC; normalizing "
                "defensively. discard_ranges may be slightly misaligned.",
                stacklevel=3,
            )
            text = new_text

    ranges = _normalize_ranges(
        raw_ranges,
        text=text,
        range_units=range_units,
        item_id=item_id,
    )
    return LabeledArticle(
        item_id=item_id,
        content_sha256=sha,
        markdown_text=text,
        discard_ranges=tuple(ranges),
    )


def _normalize_ranges(
    raw: Iterable[Sequence[int]],
    *,
    text: str,
    range_units: RangeUnits,
    item_id: str,
) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    for r in raw:
        if len(r) != 2:
            raise ValueError(
                f"item_id={item_id}: each discard range must be [start, stop]; got {r!r}"
            )
        pairs.append((int(r[0]), int(r[1])))

    if range_units == "byte":
        pairs = _convert_byte_ranges_to_codepoint(text, pairs, item_id=item_id)

    _validate_ranges(pairs, text=text, item_id=item_id)
    return pairs


def _convert_byte_ranges_to_codepoint(
    text: str,
    byte_ranges: Sequence[tuple[int, int]],
    *,
    item_id: str,
) -> list[tuple[int, int]]:
    """Map byte offsets into UTF-8 to codepoint offsets in `text`.

    Builds a `byte_offset -> codepoint_index` lookup covering only valid
    codepoint boundaries. A range whose start or stop falls inside a
    multibyte codepoint is rejected as an invalid label.
    """
    byte_to_cp: dict[int, int] = {0: 0}
    byte_pos = 0
    for cp_idx, ch in enumerate(text):
        byte_pos += len(ch.encode("utf-8"))
        byte_to_cp[byte_pos] = cp_idx + 1

    out: list[tuple[int, int]] = []
    for s, e in byte_ranges:
        if s not in byte_to_cp or e not in byte_to_cp:
            raise ValueError(
                f"item_id={item_id}: byte range [{s}, {e}) does not align "
                "to UTF-8 codepoint boundaries"
            )
        out.append((byte_to_cp[s], byte_to_cp[e]))
    return out


def _validate_ranges(
    ranges: Sequence[tuple[int, int]],
    *,
    text: str,
    item_id: str,
) -> None:
    n = len(text)
    last_end = -1
    for start, stop in ranges:
        if not (0 <= start < stop <= n):
            raise ValueError(
                f"item_id={item_id}: range [{start}, {stop}) is out of bounds "
                f"or inverted (text length = {n})"
            )
        if start < last_end:
            raise ValueError(
                f"item_id={item_id}: ranges must be sorted and non-overlapping "
                f"(got start={start} after end={last_end})"
            )
        last_end = stop


def apply_discard_ranges(text: str, ranges: Sequence[tuple[int, int]]) -> str:
    """Remove each codepoint range from `text` and return the kept text.

    The round-trip companion of the loader: a `LabeledArticle`'s
    `discard_ranges` applied to its `markdown_text` should yield the
    labeler's reference "kept" string (REQ-002 acceptance criterion).
    """
    kept: list[str] = []
    cursor = 0
    for start, stop in ranges:
        kept.append(text[cursor:start])
        cursor = stop
    kept.append(text[cursor:])
    return "".join(kept)
