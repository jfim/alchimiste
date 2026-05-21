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
