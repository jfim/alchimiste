"""Read dataset identity (commit hash + dirty flag) from an oxen working tree.

The training pipeline records this on every mlflow run so the artifact has
provenance back to the exact dataset version it was trained on (REQ-014).

We invoke `oxen` as a subprocess rather than depending on a Python binding —
this matches the existing dataset-puller pattern (see alchimiste/datasets/oxen.py)
and keeps the surface area trivially small.
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

# `oxen log` prints `commit <hex>` as the first non-blank line of the entry.
# The hash is hex; observed length is 32 (oxen 0.48), but we accept any length
# of hex characters to stay forward-compatible if oxen changes its hash size.
_COMMIT_RE = re.compile(r"^commit\s+([0-9a-fA-F]+)\s*$", re.MULTILINE)

# `oxen status` says exactly this when the working tree is clean.
_CLEAN_MARKER = "nothing to commit, working tree clean"


@dataclass(frozen=True)
class OxenMeta:
    """Snapshot of an oxen working tree's identity at a point in time."""

    commit_hash: str
    dirty: bool


def read_commit(oxen_dir: Path) -> OxenMeta:
    """Read the HEAD commit hash and clean/dirty state of `oxen_dir`.

    Raises `RuntimeError` if `oxen` is not installed, if `oxen_dir` is not
    an oxen working tree, or if the log/status output cannot be parsed.
    """
    oxen_dir = Path(oxen_dir)

    try:
        log = subprocess.run(
            ["oxen", "log", "-n", "1"],
            cwd=oxen_dir,
            capture_output=True,
            check=False,
        )
    except FileNotFoundError as e:
        raise RuntimeError(
            "oxen binary not found on PATH; install oxen to read dataset metadata"
        ) from e

    if log.returncode != 0:
        raise RuntimeError(
            f"oxen log failed in {oxen_dir}: {log.stderr.decode(errors='replace').strip()}"
        )

    log_text = log.stdout.decode(errors="replace")
    match = _COMMIT_RE.search(log_text)
    if not match:
        raise RuntimeError(f"could not parse commit hash from `oxen log` output:\n{log_text}")
    commit_hash = match.group(1)

    status = subprocess.run(
        ["oxen", "status"],
        cwd=oxen_dir,
        capture_output=True,
        check=False,
    )
    if status.returncode != 0:
        raise RuntimeError(
            f"oxen status failed in {oxen_dir}: {status.stderr.decode(errors='replace').strip()}"
        )

    dirty = _CLEAN_MARKER not in status.stdout.decode(errors="replace")
    return OxenMeta(commit_hash=commit_hash, dirty=dirty)
