"""Tests for the NFC normalize helpers (REQ-013)."""

from __future__ import annotations

import unicodedata

import pytest

from alchimiste.cleaner.data.normalize import NotNFCError, assert_nfc, defensive_nfc

# Canonical NFC vs NFD example: "é" can be one codepoint (U+00E9) or
# two ("e" + combining acute, U+0301).
_NFC_E_ACUTE = "é"  # precomposed é
_NFD_E_ACUTE = "é"  # decomposed e + combining acute


def test_nfc_and_nfd_differ_at_codepoint_level() -> None:
    # Sanity: confirm the test fixtures actually have the property we
    # think they have. If this ever fails, the whole NFC contract is moot.
    assert _NFC_E_ACUTE != _NFD_E_ACUTE
    assert unicodedata.normalize("NFC", _NFD_E_ACUTE) == _NFC_E_ACUTE
    assert unicodedata.normalize("NFC", _NFC_E_ACUTE) == _NFC_E_ACUTE


def test_assert_nfc_passes_on_nfc_form() -> None:
    # Should not raise.
    assert_nfc(_NFC_E_ACUTE)
    assert_nfc("plain ASCII text")
    assert_nfc("")  # empty string is trivially NFC


def test_assert_nfc_raises_on_nfd_form() -> None:
    with pytest.raises(NotNFCError):
        assert_nfc(_NFD_E_ACUTE)


def test_defensive_nfc_round_trips_nfc_to_itself() -> None:
    assert defensive_nfc(_NFC_E_ACUTE) == _NFC_E_ACUTE
    assert defensive_nfc("plain ASCII text") == "plain ASCII text"


def test_defensive_nfc_converts_nfd_to_nfc() -> None:
    assert defensive_nfc(_NFD_E_ACUTE) == _NFC_E_ACUTE
