"""Tests for `data.loader.load_oxen_tree` (TASK-007 + TASK-008)."""

from __future__ import annotations

import hashlib
import unicodedata
from pathlib import Path

import polars as pl
import pytest

from alchimiste.cleaner.data.loader import (
    LabeledArticle,
    apply_discard_ranges,
    load_oxen_tree,
)
from alchimiste.cleaner.data.normalize import NotNFCError


def _write_tree(
    tmp_path: Path,
    rows: list[dict],
    blobs: dict[str, bytes],
    stage: str = "cleaning",
) -> Path:
    """Synthesize a minimal oxen-tree layout: <tmp>/<stage>/{rows.parquet,blobs/}."""
    stage_dir = tmp_path / stage
    blobs_dir = stage_dir / "blobs"
    blobs_dir.mkdir(parents=True)
    for sha, content in blobs.items():
        (blobs_dir / sha).write_bytes(content)
    pl.DataFrame(rows).write_parquet(stage_dir / "rows.parquet")
    return tmp_path


def _sha(text: bytes) -> str:
    return hashlib.sha256(text).hexdigest()


# --------------------------------------------------------------------------- #
# Fixture matrix                                                              #
# --------------------------------------------------------------------------- #


def _ascii_clean_fixture() -> tuple[dict, bytes, str]:
    """Clean article (no drop ranges). Kept text == full text."""
    text = "Just a normal article with no boilerplate.\n"
    body = text.encode("utf-8")
    return (
        {"item_id": "clean", "content_sha256": _sha(body), "discard_ranges": []},
        body,
        text,  # expected kept
    )


def _ascii_one_range_fixture() -> tuple[dict, bytes, str]:
    """One byte range; expected kept = everything except that span."""
    text = "Hello world. CLICK HERE. Goodbye."
    body = text.encode("utf-8")
    # Drop "CLICK HERE. " (bytes 13..25)
    s = text.index("CLICK")
    e = text.index("Goodbye")
    return (
        {"item_id": "one", "content_sha256": _sha(body), "discard_ranges": [[s, e]]},
        body,
        text[:s] + text[e:],
    )


def _ascii_multi_range_fixture() -> tuple[dict, bytes, str]:
    """Two non-overlapping byte ranges; kept skips both."""
    text = "AAA BBB CCC DDD EEE"
    body = text.encode("utf-8")
    # Drop "BBB " and "DDD "
    s1, e1 = 4, 8
    s2, e2 = 12, 16
    return (
        {
            "item_id": "multi",
            "content_sha256": _sha(body),
            "discard_ranges": [[s1, e1], [s2, e2]],
        },
        body,
        text[:s1] + text[e1:s2] + text[e2:],
    )


def _utf8_multibyte_fixture() -> tuple[dict, bytes, str]:
    """Article with multi-byte UTF-8 chars; tests byte->codepoint conversion."""
    text = "Café! Drop me. Résumé"  # "é" = 2 bytes each, NFC precomposed
    body = text.encode("utf-8")
    # Drop "Drop me. " — locate by string then convert string slice to bytes
    s_str = text.index("Drop")
    e_str = text.index("Résumé")
    s_bytes = len(text[:s_str].encode("utf-8"))
    e_bytes = len(text[:e_str].encode("utf-8"))
    return (
        {
            "item_id": "utf8",
            "content_sha256": _sha(body),
            "discard_ranges": [[s_bytes, e_bytes]],
        },
        body,
        text[:s_str] + text[e_str:],
    )


def _nfd_fixture() -> tuple[dict, bytes, str]:
    """Article whose blob is NFD (decomposed). Should fail under require_nfc=True."""
    text_nfd = unicodedata.normalize("NFD", "Café Résumé")
    body = text_nfd.encode("utf-8")
    return (
        {"item_id": "nfd", "content_sha256": _sha(body), "discard_ranges": []},
        body,
        text_nfd,  # not used; this fixture is for the error path
    )


# --------------------------------------------------------------------------- #
# Tests                                                                       #
# --------------------------------------------------------------------------- #


def test_load_clean_fixture(tmp_path: Path) -> None:
    row, body, _ = _ascii_clean_fixture()
    _write_tree(tmp_path, [row], {row["content_sha256"]: body})
    [article] = load_oxen_tree(tmp_path)
    assert isinstance(article, LabeledArticle)
    assert article.item_id == "clean"
    assert article.discard_ranges == ()


def test_load_one_range_fixture(tmp_path: Path) -> None:
    row, body, _ = _ascii_one_range_fixture()
    _write_tree(tmp_path, [row], {row["content_sha256"]: body})
    [article] = load_oxen_tree(tmp_path)
    assert len(article.discard_ranges) == 1


def test_load_multi_range_fixture(tmp_path: Path) -> None:
    row, body, _ = _ascii_multi_range_fixture()
    _write_tree(tmp_path, [row], {row["content_sha256"]: body})
    [article] = load_oxen_tree(tmp_path)
    assert len(article.discard_ranges) == 2


def test_byte_ranges_converted_to_codepoint_for_utf8(tmp_path: Path) -> None:
    row, body, expected_kept = _utf8_multibyte_fixture()
    _write_tree(tmp_path, [row], {row["content_sha256"]: body})
    [article] = load_oxen_tree(tmp_path, range_units="byte")
    # The on-the-wire bytes were e.g. 6..16 but in codepoints they should
    # be smaller because "é" collapses 2 bytes -> 1 codepoint.
    s, _e = article.discard_ranges[0]
    assert s < article.markdown_text.index("Drop") + 1  # codepoint indices
    assert apply_discard_ranges(article.markdown_text, article.discard_ranges) == expected_kept


def test_nfd_blob_raises_under_require_nfc(tmp_path: Path) -> None:
    row, body, _ = _nfd_fixture()
    _write_tree(tmp_path, [row], {row["content_sha256"]: body})
    with pytest.raises(NotNFCError):
        load_oxen_tree(tmp_path, require_nfc=True)


def test_nfd_blob_normalized_under_lax_mode(tmp_path: Path) -> None:
    row, body, _ = _nfd_fixture()
    _write_tree(tmp_path, [row], {row["content_sha256"]: body})
    with pytest.warns(UserWarning, match="not NFC"):
        [article] = load_oxen_tree(tmp_path, require_nfc=False)
    # After defensive normalization the text equals the NFC form.
    assert article.markdown_text == unicodedata.normalize("NFC", article.markdown_text)


def test_overlapping_ranges_rejected(tmp_path: Path) -> None:
    text = "abcdefghij"
    body = text.encode("utf-8")
    row = {
        "item_id": "bad",
        "content_sha256": _sha(body),
        "discard_ranges": [[0, 5], [3, 7]],  # overlap
    }
    _write_tree(tmp_path, [row], {row["content_sha256"]: body})
    with pytest.raises(ValueError, match="sorted and non-overlapping"):
        load_oxen_tree(tmp_path)


def test_out_of_bounds_range_rejected(tmp_path: Path) -> None:
    text = "abcdef"
    body = text.encode("utf-8")
    row = {
        "item_id": "bad",
        "content_sha256": _sha(body),
        "discard_ranges": [[0, 100]],
    }
    _write_tree(tmp_path, [row], {row["content_sha256"]: body})
    # In codepoint-units mode the bounds check fires directly. In
    # byte-units mode the byte->codepoint conversion catches it first
    # (byte 100 has no codepoint boundary). Either way: a ValueError.
    with pytest.raises(ValueError):
        load_oxen_tree(tmp_path, range_units="codepoint")


def test_inverted_range_rejected(tmp_path: Path) -> None:
    text = "abcdef"
    body = text.encode("utf-8")
    row = {
        "item_id": "bad",
        "content_sha256": _sha(body),
        "discard_ranges": [[5, 5]],  # stop == start, not > start
    }
    _write_tree(tmp_path, [row], {row["content_sha256"]: body})
    with pytest.raises(ValueError, match="out of bounds or inverted"):
        load_oxen_tree(tmp_path)


def test_byte_offset_mid_codepoint_rejected(tmp_path: Path) -> None:
    """A byte offset that lands inside a multibyte char must be flagged."""
    text = "Café"  # "é" = 2 bytes at byte offset 3..5
    body = text.encode("utf-8")
    row = {
        "item_id": "bad",
        "content_sha256": _sha(body),
        # Byte 4 is the middle of "é" — invalid boundary.
        "discard_ranges": [[3, 4]],
    }
    _write_tree(tmp_path, [row], {row["content_sha256"]: body})
    with pytest.raises(ValueError, match="codepoint boundaries"):
        load_oxen_tree(tmp_path, range_units="byte")


def test_missing_blob_raises(tmp_path: Path) -> None:
    row = {
        "item_id": "missing",
        "content_sha256": "deadbeef" * 8,
        "discard_ranges": [],
    }
    _write_tree(tmp_path, [row], {})  # blob not written
    with pytest.raises(FileNotFoundError, match="missing blob"):
        load_oxen_tree(tmp_path)


def test_missing_parquet_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match=r"missing rows\.parquet"):
        load_oxen_tree(tmp_path)


# --------------------------------------------------------------------------- #
# TASK-008: round-trip                                                        #
# --------------------------------------------------------------------------- #

ROUND_TRIP_FIXTURES = [
    _ascii_clean_fixture(),
    _ascii_one_range_fixture(),
    _ascii_multi_range_fixture(),
    _utf8_multibyte_fixture(),
]


@pytest.mark.parametrize(
    "row,body,expected_kept",
    ROUND_TRIP_FIXTURES,
    ids=["clean", "one_range", "multi_range", "utf8_multibyte"],
)
def test_round_trip(tmp_path: Path, row: dict, body: bytes, expected_kept: str) -> None:
    _write_tree(tmp_path, [row], {row["content_sha256"]: body})
    [article] = load_oxen_tree(tmp_path)
    kept = apply_discard_ranges(article.markdown_text, article.discard_ranges)
    assert kept == expected_kept


def test_round_trip_loads_many_fixtures_at_once(tmp_path: Path) -> None:
    """REQ-002 round-trip on multiple fixtures simultaneously (>= 5)."""
    fixtures = [
        _ascii_clean_fixture(),
        _ascii_one_range_fixture(),
        _ascii_multi_range_fixture(),
        _utf8_multibyte_fixture(),
        # Add a fifth fixture: a "drop the whole thing" article.
        (
            {
                "item_id": "drop_all",
                "content_sha256": _sha(b"all-boilerplate"),
                "discard_ranges": [[0, len(b"all-boilerplate")]],
            },
            b"all-boilerplate",
            "",
        ),
    ]
    rows = [f[0] for f in fixtures]
    blobs = {f[0]["content_sha256"]: f[1] for f in fixtures}
    expected = {f[0]["item_id"]: f[2] for f in fixtures}
    _write_tree(tmp_path, rows, blobs)
    articles = load_oxen_tree(tmp_path)
    assert len(articles) == 5
    for a in articles:
        kept = apply_discard_ranges(a.markdown_text, a.discard_ranges)
        assert kept == expected[a.item_id], f"round-trip failed for {a.item_id}"
