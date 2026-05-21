# Alchimiste Dataset Puller Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Pull a fresh, content-addressed dataset (per stage) from alambic and commit it into an oxen repository, including all referenced blobs, with sha256 verification on every blob fetched.

**Architecture:** A `alchimiste pull` CLI command that calls alambic's `/api/datasets/:stage/rows.parquet` once, walks the referenced `content_sha256` values, downloads only blobs that are not already in the local oxen working tree (`blobs/<sha>.gz` is the local layout), verifies each blob's hash on receipt, prunes orphan blobs, then runs `oxen add` + `oxen commit`. Oxen is invoked via subprocess on the `oxen` CLI binary — keeps the integration trivial and lets the user retain control of repo init / remotes.

**Tech Stack:** Python ≥ 3.12, uv, [polars](https://pola.rs/) for parquet, [httpx](https://www.python-httpx.org/) for HTTP, stdlib `hashlib` / `gzip`, `oxen` CLI as a subprocess.

**Out of scope:**
- Training code (separate plan, separate session).
- Auth (alambic side hasn't shipped it yet).
- Concurrent blob fetches (sequential is fine at <100 blobs/pull at expected volume).
- mlflow integration (separate plan).

---

## File Structure

**New files:**
- `src/alchimiste/datasets/__init__.py` — empty init.
- `src/alchimiste/datasets/client.py` — alambic HTTP client.
- `src/alchimiste/datasets/sync.py` — blob sync + orphan detection.
- `src/alchimiste/datasets/oxen.py` — subprocess wrapper.
- `src/alchimiste/cli.py` — typer/argparse entry point.
- `tests/datasets/test_client.py`
- `tests/datasets/test_sync.py`
- `tests/datasets/test_cli.py`

**Modified files:**
- `pyproject.toml` — add deps + script entry point.
- `justfile` — add `just pull-extraction <oxen-dir>` and `just pull-cleaning <oxen-dir>` recipes (optional convenience).

---

## Task 1: Add dependencies and CLI entry point

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Add runtime deps**

Edit `pyproject.toml`. Replace `dependencies = []` with:

```toml
dependencies = [
    "httpx>=0.27",
    "polars>=1.0",
]
```

In dev deps section (`[dependency-groups]` → `dev`), add `pytest-httpx`:

```toml
dev = [
    "ruff>=0.1",
    "pytest>=8",
    "pytest-httpx>=0.30",
]
```

Add a script entry point:

```toml
[project.scripts]
alchimiste = "alchimiste.cli:main"
```

- [ ] **Step 2: Sync and verify**

Run: `just sync`
Expected: deps install, `which alchimiste || uv run alchimiste --help` shows a help message (after Task 5 wires the CLI; for now `uv run python -c "import httpx, polars"` should succeed).

Run: `uv run python -c "import httpx, polars; print('ok')"`
Expected: `ok`.

- [ ] **Step 3: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "deps: add httpx, polars, pytest-httpx for dataset puller"
```

---

## Task 2: Alambic HTTP client

A small, testable wrapper. Two methods:
- `fetch_rows(stage) -> bytes` (raw parquet)
- `fetch_blob(stage, sha256) -> bytes` (raw, with hash verified)

**Files:**
- Create: `src/alchimiste/datasets/__init__.py` (empty)
- Create: `src/alchimiste/datasets/client.py`
- Create: `tests/__init__.py` (empty)
- Create: `tests/datasets/__init__.py` (empty)
- Create: `tests/datasets/test_client.py`

- [ ] **Step 1: Write failing test**

Create `tests/datasets/test_client.py`:

```python
import hashlib

import pytest

from alchimiste.datasets.client import AlambicClient, BlobHashMismatch


def test_fetch_rows_returns_bytes(httpx_mock):
    httpx_mock.add_response(
        url="http://alambic.test/api/datasets/extraction/rows.parquet",
        content=b"PAR1\x00binary",
    )
    client = AlambicClient("http://alambic.test")
    assert client.fetch_rows("extraction") == b"PAR1\x00binary"


def test_fetch_blob_verifies_hash(httpx_mock):
    payload = b"<html></html>"
    sha = hashlib.sha256(payload).hexdigest()
    httpx_mock.add_response(
        url=f"http://alambic.test/api/datasets/extraction/blobs/{sha}",
        content=payload,
    )
    client = AlambicClient("http://alambic.test")
    assert client.fetch_blob("extraction", sha) == payload


def test_fetch_blob_raises_on_hash_mismatch(httpx_mock):
    sha = "0" * 64
    httpx_mock.add_response(
        url=f"http://alambic.test/api/datasets/extraction/blobs/{sha}",
        content=b"different bytes",
    )
    client = AlambicClient("http://alambic.test")
    with pytest.raises(BlobHashMismatch):
        client.fetch_blob("extraction", sha)


def test_fetch_blob_404_raises(httpx_mock):
    sha = "1" * 64
    httpx_mock.add_response(
        url=f"http://alambic.test/api/datasets/extraction/blobs/{sha}",
        status_code=404,
    )
    client = AlambicClient("http://alambic.test")
    with pytest.raises(FileNotFoundError):
        client.fetch_blob("extraction", sha)
```

Run: `uv run pytest tests/datasets/test_client.py -v`
Expected: FAIL (import error — module missing).

- [ ] **Step 2: Implement the client**

Create `src/alchimiste/datasets/__init__.py` (empty).

Create `src/alchimiste/datasets/client.py`:

```python
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
```

Create `tests/__init__.py` and `tests/datasets/__init__.py` as empty files.

- [ ] **Step 3: Run tests**

Run: `uv run pytest tests/datasets/test_client.py -v`
Expected: 4 tests pass.

- [ ] **Step 4: Commit**

```bash
git add src/alchimiste/datasets/__init__.py src/alchimiste/datasets/client.py tests/__init__.py tests/datasets/__init__.py tests/datasets/test_client.py
git commit -m "feat: alambic dataset HTTP client"
```

---

## Task 3: Blob sync — missing detection, fetch, prune

Given a target directory (`blobs/`), a list of required sha256 hashes, and a client, sync:
1. Determine which referenced hashes are missing locally.
2. Fetch each missing blob and write it to `blobs/<sha>` (raw bytes; gzip is alambic's on-disk concern only — the HTTP API returns raw).
3. Determine orphans (files in `blobs/` not referenced by any row).
4. Delete orphans.

Returns a small summary dict.

**Files:**
- Create: `src/alchimiste/datasets/sync.py`
- Create: `tests/datasets/test_sync.py`

- [ ] **Step 1: Write failing test**

Create `tests/datasets/test_sync.py`:

```python
import hashlib
from pathlib import Path

from alchimiste.datasets.sync import sync_blobs


class FakeClient:
    def __init__(self, store):
        self.store = store
        self.fetched = []

    def fetch_blob(self, stage, sha):
        self.fetched.append(sha)
        return self.store[sha]


def test_sync_fetches_only_missing(tmp_path: Path):
    payload_a = b"alpha"
    payload_b = b"beta"
    sha_a = hashlib.sha256(payload_a).hexdigest()
    sha_b = hashlib.sha256(payload_b).hexdigest()

    blobs_dir = tmp_path / "blobs"
    blobs_dir.mkdir()
    (blobs_dir / sha_a).write_bytes(payload_a)

    client = FakeClient({sha_b: payload_b})
    summary = sync_blobs(client, "extraction", blobs_dir, required={sha_a, sha_b})

    assert client.fetched == [sha_b]
    assert (blobs_dir / sha_b).read_bytes() == payload_b
    assert summary == {"fetched": 1, "pruned": 0, "total_required": 2}


def test_sync_prunes_orphans(tmp_path: Path):
    payload = b"keep"
    sha = hashlib.sha256(payload).hexdigest()
    orphan_sha = "0" * 64

    blobs_dir = tmp_path / "blobs"
    blobs_dir.mkdir()
    (blobs_dir / sha).write_bytes(payload)
    (blobs_dir / orphan_sha).write_bytes(b"stale")

    client = FakeClient({})
    summary = sync_blobs(client, "extraction", blobs_dir, required={sha})

    assert not (blobs_dir / orphan_sha).exists()
    assert (blobs_dir / sha).exists()
    assert summary == {"fetched": 0, "pruned": 1, "total_required": 1}


def test_sync_creates_blobs_dir_if_missing(tmp_path: Path):
    payload = b"x"
    sha = hashlib.sha256(payload).hexdigest()
    blobs_dir = tmp_path / "blobs"

    client = FakeClient({sha: payload})
    sync_blobs(client, "extraction", blobs_dir, required={sha})

    assert (blobs_dir / sha).read_bytes() == payload
```

Run: `uv run pytest tests/datasets/test_sync.py -v`
Expected: FAIL.

- [ ] **Step 2: Implement sync**

Create `src/alchimiste/datasets/sync.py`:

```python
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
```

- [ ] **Step 3: Run tests**

Run: `uv run pytest tests/datasets/test_sync.py -v`
Expected: 3 tests pass.

- [ ] **Step 4: Commit**

```bash
git add src/alchimiste/datasets/sync.py tests/datasets/test_sync.py
git commit -m "feat: blob sync with orphan pruning"
```

---

## Task 4: Oxen subprocess wrapper

Wraps `oxen add` + `oxen commit -m <msg>` invocations. Returns the resulting commit hash on success. No-ops cleanly if there's nothing to commit (oxen exits non-zero in that case; treat as success but report `None`).

**Files:**
- Create: `src/alchimiste/datasets/oxen.py`
- Create: `tests/datasets/test_oxen.py`

- [ ] **Step 1: Write failing test**

Create `tests/datasets/test_oxen.py`:

```python
from pathlib import Path
from unittest.mock import patch

import pytest

from alchimiste.datasets.oxen import oxen_commit


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

    with patch("alchimiste.datasets.oxen.subprocess.run", fake_run):
        with pytest.raises(RuntimeError):
            oxen_commit(tmp_path, "x")
```

Run: `uv run pytest tests/datasets/test_oxen.py -v`
Expected: FAIL.

- [ ] **Step 2: Implement wrapper**

Create `src/alchimiste/datasets/oxen.py`:

```python
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
```

- [ ] **Step 3: Run tests**

Run: `uv run pytest tests/datasets/test_oxen.py -v`
Expected: 3 tests pass.

- [ ] **Step 4: Commit**

```bash
git add src/alchimiste/datasets/oxen.py tests/datasets/test_oxen.py
git commit -m "feat: oxen subprocess wrapper for add + commit"
```

---

## Task 5: CLI — `alchimiste pull <stage> <repo-dir>`

Wires it together. Reads `ALAMBIC_BASE_URL` from env (or `--base-url` flag). Fetches `rows.parquet`, writes to `<repo-dir>/<stage>/rows.parquet`. Reads `content_sha256` column with polars to compute the required set. Syncs blobs into `<repo-dir>/<stage>/blobs/`. Commits.

**Files:**
- Create: `src/alchimiste/cli.py`
- Create: `tests/datasets/test_cli.py`

- [ ] **Step 1: Write failing test**

Create `tests/datasets/test_cli.py`:

```python
import hashlib
import io
from pathlib import Path
from unittest.mock import patch

import polars as pl

from alchimiste.cli import pull


def _make_parquet(rows: list[dict]) -> bytes:
    df = pl.DataFrame(rows)
    buf = io.BytesIO()
    df.write_parquet(buf)
    return buf.getvalue()


def test_pull_writes_rows_and_blobs(tmp_path: Path, httpx_mock):
    payload = b"<html></html>"
    sha = hashlib.sha256(payload).hexdigest()
    parquet = _make_parquet([
        {
            "item_id": "i1",
            "content_sha256": sha,
            "xpath": "/html",
            "confirmed_at": 1716240000,
            "updated_at": 1716240000,
            "prior_model_version": None,
        }
    ])

    httpx_mock.add_response(
        url="http://alambic.test/api/datasets/extraction/rows.parquet",
        content=parquet,
    )
    httpx_mock.add_response(
        url=f"http://alambic.test/api/datasets/extraction/blobs/{sha}",
        content=payload,
    )

    with patch("alchimiste.cli.oxen_commit", return_value="commit_abc"):
        result = pull(
            stage="extraction",
            repo_dir=tmp_path,
            base_url="http://alambic.test",
        )

    assert (tmp_path / "extraction" / "rows.parquet").read_bytes() == parquet
    assert (tmp_path / "extraction" / "blobs" / sha).read_bytes() == payload
    assert result["fetched"] == 1
    assert result["pruned"] == 0
    assert result["commit"] == "commit_abc"
```

Run: `uv run pytest tests/datasets/test_cli.py -v`
Expected: FAIL (import / function missing).

- [ ] **Step 2: Implement CLI**

Create `src/alchimiste/cli.py`:

```python
"""CLI entry point for alchimiste."""

from __future__ import annotations

import argparse
import io
import os
import sys
from pathlib import Path
from typing import Any

import polars as pl

from alchimiste.datasets.client import AlambicClient
from alchimiste.datasets.oxen import oxen_commit
from alchimiste.datasets.sync import sync_blobs

STAGES = ("extraction", "cleaning")


def pull(
    stage: str,
    repo_dir: Path,
    base_url: str,
    skip_commit: bool = False,
) -> dict[str, Any]:
    if stage not in STAGES:
        raise ValueError(f"unknown stage {stage!r}; expected one of {STAGES}")

    stage_dir = repo_dir / stage
    stage_dir.mkdir(parents=True, exist_ok=True)
    blobs_dir = stage_dir / "blobs"

    client = AlambicClient(base_url)
    try:
        parquet_bytes = client.fetch_rows(stage)
        (stage_dir / "rows.parquet").write_bytes(parquet_bytes)

        df = pl.read_parquet(io.BytesIO(parquet_bytes))
        required = set(df["content_sha256"].to_list())

        summary = sync_blobs(client, stage, blobs_dir, required=required)
    finally:
        client.close()

    commit = None
    if not skip_commit:
        commit = oxen_commit(repo_dir, f"pull {stage} n={summary['total_required']}")

    return {**summary, "commit": commit}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="alchimiste")
    sub = parser.add_subparsers(dest="cmd", required=True)

    pull_p = sub.add_parser("pull", help="pull a dataset stage from alambic into an oxen repo")
    pull_p.add_argument("stage", choices=STAGES)
    pull_p.add_argument("repo_dir", type=Path)
    pull_p.add_argument(
        "--base-url",
        default=os.environ.get("ALAMBIC_BASE_URL"),
        help="alambic base URL (or set ALAMBIC_BASE_URL env)",
    )
    pull_p.add_argument("--skip-commit", action="store_true")

    args = parser.parse_args(argv)
    if args.cmd == "pull":
        if not args.base_url:
            parser.error("--base-url or ALAMBIC_BASE_URL required")
        result = pull(
            stage=args.stage,
            repo_dir=args.repo_dir,
            base_url=args.base_url,
            skip_commit=args.skip_commit,
        )
        print(result)
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 3: Run tests**

Run: `uv run pytest tests/datasets/test_cli.py -v`
Expected: 1 test passes.

- [ ] **Step 4: Smoke the CLI**

Run: `uv run alchimiste pull --help`
Expected: usage shown.

- [ ] **Step 5: Commit**

```bash
git add src/alchimiste/cli.py tests/datasets/test_cli.py
git commit -m "feat: alchimiste pull CLI"
```

---

## Task 6: Optional justfile recipes + final sweep

**Files:**
- Modify: `justfile`

- [ ] **Step 1: Add convenience recipes**

Append to `justfile`:

```make
pull-extraction repo_dir:
    uv run alchimiste pull extraction {{repo_dir}}

pull-cleaning repo_dir:
    uv run alchimiste pull cleaning {{repo_dir}}
```

- [ ] **Step 2: Full check**

Run: `just check`
Expected: lint + format + tests all green.

- [ ] **Step 3: Commit**

```bash
git add justfile
git commit -m "chore: justfile pull recipes"
```

---

## Self-Review Checklist

- HTTP client covered (rows + blob with hash verification) ✓.
- Blob sync covers fetch-missing and prune-orphans ✓.
- Oxen invocation present and tolerates "nothing to commit" ✓.
- CLI ties it together; tested end-to-end with httpx_mock ✓.
- No training code (deliberately deferred).
- All blob types named by sha256 of raw bytes (alambic-side gzip is invisible to the consumer).
- `polars` chosen over `pyarrow` to stay aligned with session 1's tooling decisions.
