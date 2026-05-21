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
    parquet = _make_parquet(
        [
            {
                "item_id": "i1",
                "content_sha256": sha,
                "xpath": "/html",
                "confirmed_at": 1716240000,
                "updated_at": 1716240000,
                "prior_model_version": None,
            }
        ]
    )

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
