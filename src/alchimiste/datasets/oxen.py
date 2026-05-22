"""Thin wrapper around the `oxen` CLI."""

from __future__ import annotations

import subprocess
from pathlib import Path


def oxen_commit(repo_dir: Path, message: str) -> str | None:
    """Run `oxen add . && oxen commit -m <message>` in `repo_dir`.

    Returns the commit hash from stdout on success, or `None` if there was
    nothing to commit. Raises `RuntimeError` on any other failure.
    """
    add = subprocess.run(
        ["oxen", "add", "."],
        cwd=repo_dir,
        capture_output=True,
    )
    if add.returncode != 0:
        raise RuntimeError(f"oxen add failed: {add.stderr.decode(errors='replace')}")

    commit = subprocess.run(
        ["oxen", "commit", "-m", message],
        cwd=repo_dir,
        capture_output=True,
    )
    if commit.returncode == 0:
        return commit.stdout.decode().strip() or None
    stderr = commit.stderr.decode(errors="replace").lower()
    if "nothing to commit" in stderr or "no changes" in stderr:
        return None
    raise RuntimeError(f"oxen commit failed: {commit.stderr.decode(errors='replace')}")


def oxen_push(repo_dir: Path) -> None:
    """Run `oxen push` in `repo_dir`.

    Pushes the current branch to the default remote. Raises `RuntimeError`
    on failure; the caller decides whether to block or warn.
    """
    push = subprocess.run(
        ["oxen", "push"],
        cwd=repo_dir,
        capture_output=True,
    )
    if push.returncode != 0:
        raise RuntimeError(f"oxen push failed: {push.stderr.decode(errors='replace')}")
