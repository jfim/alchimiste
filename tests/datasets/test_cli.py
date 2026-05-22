import hashlib
import io
from pathlib import Path
from unittest.mock import patch

import polars as pl

from alchimiste import cli as cli_module
from alchimiste.cli import _load_local_config_value, pull


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

    with (
        patch("alchimiste.cli.oxen_commit", return_value="commit_abc"),
        patch("alchimiste.cli.oxen_push") as mock_push,
    ):
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
    assert result["pushed"] is True
    mock_push.assert_called_once_with(tmp_path)


def _register_one_row(httpx_mock) -> bytes:
    """Stand up a single-row alambic response so `pull()` has something to sync."""
    payload = b"<html></html>"
    sha = hashlib.sha256(payload).hexdigest()
    parquet = _make_parquet(
        [{"item_id": "i1", "content_sha256": sha, "xpath": "/html"}]
    )
    httpx_mock.add_response(
        url="http://alambic.test/api/datasets/extraction/rows.parquet",
        content=parquet,
    )
    httpx_mock.add_response(
        url=f"http://alambic.test/api/datasets/extraction/blobs/{sha}",
        content=payload,
    )
    return parquet


def test_pull_skips_push_when_disabled(tmp_path: Path, httpx_mock):
    _register_one_row(httpx_mock)

    with (
        patch("alchimiste.cli.oxen_commit", return_value="commit_abc"),
        patch("alchimiste.cli.oxen_push") as mock_push,
    ):
        result = pull(
            stage="extraction",
            repo_dir=tmp_path,
            base_url="http://alambic.test",
            push=False,
        )

    mock_push.assert_not_called()
    assert result["pushed"] is False


def test_pull_skips_push_when_no_new_commit(tmp_path: Path, httpx_mock):
    _register_one_row(httpx_mock)

    # No-op pull (nothing changed) → oxen_commit returns None → no push.
    with (
        patch("alchimiste.cli.oxen_commit", return_value=None),
        patch("alchimiste.cli.oxen_push") as mock_push,
    ):
        result = pull(
            stage="extraction",
            repo_dir=tmp_path,
            base_url="http://alambic.test",
        )

    mock_push.assert_not_called()
    assert result["commit"] is None
    assert result["pushed"] is False


def test_load_local_config_value_returns_none_when_absent(tmp_path: Path):
    with patch.object(cli_module, "_LOCAL_CONFIG_PATH", tmp_path / "nope.yaml"):
        assert _load_local_config_value("alambic", "base_url") is None


def test_load_local_config_value_reads_nested_key(tmp_path: Path):
    local = tmp_path / "local.yaml"
    local.write_text(
        "alambic:\n  base_url: https://alambic.test\n",
        encoding="utf-8",
    )
    with patch.object(cli_module, "_LOCAL_CONFIG_PATH", local):
        assert _load_local_config_value("alambic", "base_url") == "https://alambic.test"


def test_load_local_config_value_returns_none_for_missing_key(tmp_path: Path):
    local = tmp_path / "local.yaml"
    local.write_text("mlflow:\n  tracking_uri: https://x\n", encoding="utf-8")
    with patch.object(cli_module, "_LOCAL_CONFIG_PATH", local):
        # alambic section not present at all
        assert _load_local_config_value("alambic", "base_url") is None
        # half-present chain
        assert _load_local_config_value("mlflow", "missing") is None
