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
