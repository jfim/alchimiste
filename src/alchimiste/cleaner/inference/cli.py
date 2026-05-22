"""stdin -> JSON stdout CLI for the cleaner (TASK-026, TASK-027, REQ-012).

Wired to `just predict artifact=runs/<ts>`. Reads a UTF-8 markdown
document from stdin, runs `predict_text`, and emits
`{"drop_ranges": [[start, stop], ...]}` on stdout.

`--bench` (TASK-027) times one forward pass over the input and prints
the wall time. Fails non-zero (with a clear message) if the wall time
exceeds 300 seconds (NFR-003 hard cap).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from alchimiste.cleaner.inference.pyfunc import predict_text

# NFR-003: inference shall complete in <= 300 s per article on a modern
# laptop CPU. The bench mode enforces this; the normal predict path
# only flags it as a warning so a slow input doesn't break a pipeline.
_BENCH_HARD_CAP_SECONDS = 300.0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="alchimiste-clean",
        description=(
            "Read UTF-8 markdown from stdin and emit drop_ranges JSON. "
            "Codepoint offsets in the NFC-normalized input."
        ),
    )
    parser.add_argument(
        "--artifact",
        required=True,
        type=Path,
        help="Path to the artifact directory (config.yaml + model/ + threshold.json).",
    )
    parser.add_argument(
        "--bench",
        action="store_true",
        help="Print elapsed wall time to stderr; exit non-zero if > 300s.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    text = sys.stdin.read()

    start = time.perf_counter()
    result = predict_text(args.artifact, text)
    elapsed = time.perf_counter() - start

    sys.stdout.write(json.dumps(result, ensure_ascii=False) + "\n")

    if args.bench:
        sys.stderr.write(f"alchimiste-clean: elapsed_seconds={elapsed:.3f}\n")
        if elapsed > _BENCH_HARD_CAP_SECONDS:
            sys.stderr.write(
                f"alchimiste-clean: exceeded NFR-003 cap of {_BENCH_HARD_CAP_SECONDS:.0f}s\n"
            )
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
