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
from alchimiste.datasets.oxen import oxen_commit, oxen_push
from alchimiste.datasets.sync import sync_blobs

STAGES = ("extraction", "cleaning")


def pull(
    stage: str,
    repo_dir: Path,
    base_url: str,
    skip_commit: bool = False,
    push: bool = True,
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
    pushed = False
    if not skip_commit:
        commit = oxen_commit(repo_dir, f"pull {stage} n={summary['total_required']}")
        # Only push when there was actually a new commit. If `commit` is
        # None there's nothing to publish, and `oxen push` on a no-op
        # state still hits the remote — skip it.
        if push and commit is not None:
            oxen_push(repo_dir)
            pushed = True

    return {**summary, "commit": commit, "pushed": pushed}


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
    pull_p.add_argument(
        "--no-push",
        action="store_true",
        help="commit locally but skip `oxen push` (default: push after commit)",
    )

    args = parser.parse_args(argv)
    if args.cmd == "pull":
        if not args.base_url:
            parser.error("--base-url or ALAMBIC_BASE_URL required")
        result = pull(
            stage=args.stage,
            repo_dir=args.repo_dir,
            base_url=args.base_url,
            skip_commit=args.skip_commit,
            push=not args.no_push,
        )
        print(result)
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
