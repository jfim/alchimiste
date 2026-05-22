from pathlib import Path
from unittest.mock import patch

import pytest

from alchimiste.datasets.oxen import oxen_commit, oxen_push


def _runs(cmds: list[list[str]]):
    """Build a fake subprocess.run that records and returns canned outputs."""
    calls: list[list[str]] = []

    def fake_run(args, **kwargs):
        calls.append(args)
        # Return last-arg-conditional responses
        cmd = args[1] if len(args) > 1 else ""

        class R:
            returncode = 0
            stdout = b"abc123commit\n" if cmd == "commit" else b""
            stderr = b""

        return R()

    return calls, fake_run


def test_oxen_commit_invokes_add_then_commit(tmp_path: Path):
    calls, fake_run = _runs([])
    with patch("alchimiste.datasets.oxen.subprocess.run", fake_run):
        oxen_commit(tmp_path, "test message")

    assert calls[0][:2] == ["oxen", "add"]
    assert calls[1][:2] == ["oxen", "commit"]
    assert "test message" in calls[1]


def test_oxen_commit_nothing_to_commit_returns_none(tmp_path: Path):
    def fake_run(args, **kwargs):
        class R:
            returncode = 1 if args[1] == "commit" else 0
            stdout = b""
            stderr = b"Nothing to commit\n"

        return R()

    with patch("alchimiste.datasets.oxen.subprocess.run", fake_run):
        assert oxen_commit(tmp_path, "x") is None


def test_oxen_commit_raises_on_other_failure(tmp_path: Path):
    def fake_run(args, **kwargs):
        class R:
            returncode = 2
            stdout = b""
            stderr = b"some other oxen error\n"

        return R()

    with (
        patch("alchimiste.datasets.oxen.subprocess.run", fake_run),
        pytest.raises(RuntimeError),
    ):
        oxen_commit(tmp_path, "x")


def test_oxen_push_invokes_oxen_push(tmp_path: Path):
    calls: list[list[str]] = []

    def fake_run(args, **kwargs):
        calls.append(args)

        class R:
            returncode = 0
            stdout = b""
            stderr = b""

        return R()

    with patch("alchimiste.datasets.oxen.subprocess.run", fake_run):
        oxen_push(tmp_path)

    assert calls == [["oxen", "push"]]


def test_oxen_push_raises_on_failure(tmp_path: Path):
    def fake_run(args, **kwargs):
        class R:
            returncode = 1
            stdout = b""
            stderr = b"remote unreachable\n"

        return R()

    with (
        patch("alchimiste.datasets.oxen.subprocess.run", fake_run),
        pytest.raises(RuntimeError, match="remote unreachable"),
    ):
        oxen_push(tmp_path)
