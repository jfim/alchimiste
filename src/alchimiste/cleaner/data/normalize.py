"""Unicode NFC normalization helpers (REQ-013, design.md § 2.3).

The text cleaner's contract is **codepoint offsets into NFC-normalized text**.
This module provides the two operations the loader and inference paths need:

* `assert_nfc` — verify input is already NFC; raise otherwise. Used by the
  loader when `data.require_nfc=true` to fail loudly on label-shifting
  normalization mismatches.
* `defensive_nfc` — always return the NFC form. Used by the inference
  pyfunc to be tolerant of callers who didn't normalize, and by the loader
  in transitional mode (`data.require_nfc=false`).
"""

from __future__ import annotations

import unicodedata


class NotNFCError(ValueError):
    """Raised when text is required to be in NFC but isn't."""


def assert_nfc(text: str) -> None:
    """Raise `NotNFCError` if `text` is not already in Unicode NFC form.

    Cheap structural check: re-normalize and compare. The two are byte-identical
    iff `text` is already NFC, since NFC is idempotent.
    """
    if unicodedata.normalize("NFC", text) != text:
        raise NotNFCError("text is not in Unicode NFC form")


def defensive_nfc(text: str) -> str:
    """Return the NFC form of `text`. Always succeeds (no error path)."""
    return unicodedata.normalize("NFC", text)
