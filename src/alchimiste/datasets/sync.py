"""Blob synchronization: fetch missing, prune orphans."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol


class _BlobFetcher(Protocol):
    def fetch_blob(self, stage: str, sha256: str) -> bytes: ...


def sync_blobs(
    client: _BlobFetcher,
    stage: str,
    blobs_dir: Path,
    required: set[str],
) -> dict[str, int]:
    """Ensure exactly the `required` blobs exist under `blobs_dir`.

    Returns a summary: {"fetched": int, "pruned": int, "total_required": int}.
    """
    blobs_dir.mkdir(parents=True, exist_ok=True)
    existing = {p.name for p in blobs_dir.iterdir() if p.is_file()}

    missing = required - existing
    orphans = existing - required

    for sha in sorted(missing):
        data = client.fetch_blob(stage, sha)
        (blobs_dir / sha).write_bytes(data)

    for sha in sorted(orphans):
        (blobs_dir / sha).unlink()

    return {
        "fetched": len(missing),
        "pruned": len(orphans),
        "total_required": len(required),
    }
