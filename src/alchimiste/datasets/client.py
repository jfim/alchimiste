"""HTTP client for alambic's dataset endpoints."""

from __future__ import annotations

import hashlib

import httpx


class BlobHashMismatch(Exception):
    """Raised when a fetched blob's sha256 does not match the requested key."""


class AlambicClient:
    def __init__(self, base_url: str, timeout: float = 60.0) -> None:
        self._base = base_url.rstrip("/")
        self._client = httpx.Client(timeout=timeout)

    def fetch_rows(self, stage: str) -> bytes:
        resp = self._client.get(f"{self._base}/api/datasets/{stage}/rows.parquet")
        resp.raise_for_status()
        return resp.content

    def fetch_blob(self, stage: str, sha256: str) -> bytes:
        resp = self._client.get(f"{self._base}/api/datasets/{stage}/blobs/{sha256}")
        if resp.status_code == 404:
            raise FileNotFoundError(sha256)
        resp.raise_for_status()
        body = resp.content
        actual = hashlib.sha256(body).hexdigest()
        if actual != sha256:
            raise BlobHashMismatch(f"expected {sha256}, got {actual}")
        return body

    def close(self) -> None:
        self._client.close()
