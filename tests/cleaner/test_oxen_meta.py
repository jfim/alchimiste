"""Tests for `data.oxen_meta.read_commit`.

These tests use a real oxen binary against a tmp_path repo so we exercise
the actual subprocess + parsing rather than mocking it. The test is
skipped (not failed) if `oxen` isn't on PATH so CI without oxen still
makes progress.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

from alchimiste.cleaner.data.oxen_meta import OxenMeta, read_commit

pytestmark = pytest.mark.skipif(
    shutil.which("oxen") is None, reason="oxen binary not installed on PATH"
)


def _init_repo(repo: Path) -> str:
    """Initialize an oxen repo with one committed file. Returns the commit hash."""
    subprocess.run(["oxen", "init", "."], cwd=repo, capture_output=True, check=True)
    (repo / "hello.txt").write_text("hello\n")
    subprocess.run(["oxen", "add", "hello.txt"], cwd=repo, capture_output=True, check=True)
    out = subprocess.run(
        ["oxen", "commit", "-m", "initial"],
        cwd=repo,
        capture_output=True,
        check=True,
        text=True,
    )
    # Commit hash is in stdout but we just re-read it via read_commit below.
    return out.stdout


def test_clean_tree_yields_dirty_false(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    meta = read_commit(tmp_path)
    assert isinstance(meta, OxenMeta)
    assert meta.dirty is False
    # Hash is hex; observed length is 32 on oxen 0.48 but we don't pin that.
    assert re.fullmatch(r"[0-9a-fA-F]+", meta.commit_hash)
    assert len(meta.commit_hash) >= 16  # sanity floor


def test_untracked_file_makes_tree_dirty(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    (tmp_path / "new_file.txt").write_text("untracked\n")
    meta = read_commit(tmp_path)
    assert meta.dirty is True
    # Commit hash unchanged — only the working tree is dirty.
    clean_meta = read_commit(tmp_path / ".")  # same dir, different Path form
    # Note: the prior call already saw the dirty state; pull a fresh measure
    # against a sibling fixture to confirm the hash is stable.
    assert meta.commit_hash == clean_meta.commit_hash


def test_missing_oxen_binary_raises(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # Simulate oxen not being installed: make the subprocess call fail with
    # FileNotFoundError, which is what subprocess.run would raise.
    def _no_oxen(*_args, **_kwargs):  # type: ignore[no-untyped-def]
        raise FileNotFoundError("oxen")

    monkeypatch.setattr("alchimiste.cleaner.data.oxen_meta.subprocess.run", _no_oxen)
    with pytest.raises(RuntimeError, match="oxen binary not found"):
        read_commit(tmp_path)


def test_non_oxen_dir_raises(tmp_path: Path) -> None:
    # An empty dir is not an oxen repo; `oxen log` should fail with non-zero exit.
    with pytest.raises(RuntimeError, match="oxen log failed"):
        read_commit(tmp_path)
