"""Print summary statistics for a cleaning corpus (TASK-011).

Invoked as a Hydra entrypoint via `just label-stats [data.oxen_dir=...]`.
Reports the four numbers a labeler / trainer typically wants:

  * total article count
  * fraction with empty `discard_ranges` (the "clean" majority — REQ-003)
  * distribution of range counts per article (min/mean/median/max)
  * distribution of range lengths in codepoints

The output goes to stdout in a plain key:value format so it's easy to
eyeball and grep but also machine-readable enough for a smoke test.
"""

from __future__ import annotations

import statistics
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig

from alchimiste.cleaner.data.loader import LabeledArticle, load_oxen_tree

# Resolve `configs/` relative to this file so the entrypoint works
# regardless of cwd. The package layout is
#   src/alchimiste/cleaner/data/label_stats.py
#                                 ^             this file
# and the configs live at:
#   configs/                                     5 levels up
_CONFIG_PATH = str(Path(__file__).resolve().parents[4] / "configs")


def compute_stats(articles: list[LabeledArticle]) -> dict[str, object]:
    """Pure-function half of the entrypoint — easy to unit-test."""
    n = len(articles)
    if n == 0:
        return {
            "total_articles": 0,
            "clean_fraction": 0.0,
            "range_counts": {"min": 0, "mean": 0.0, "median": 0.0, "max": 0},
            "range_lengths": {"min": 0, "mean": 0.0, "median": 0.0, "max": 0},
        }

    range_counts = [len(a.discard_ranges) for a in articles]
    range_lengths = [stop - start for a in articles for (start, stop) in a.discard_ranges]
    n_clean = sum(1 for c in range_counts if c == 0)

    def _summary(values: list[int]) -> dict[str, float]:
        if not values:
            return {"min": 0, "mean": 0.0, "median": 0.0, "max": 0}
        return {
            "min": min(values),
            "mean": statistics.fmean(values),
            "median": statistics.median(values),
            "max": max(values),
        }

    return {
        "total_articles": n,
        "clean_fraction": n_clean / n,
        "range_counts": _summary(range_counts),
        "range_lengths": _summary(range_lengths),
    }


def render_stats(stats: dict[str, object]) -> str:
    """Format `compute_stats` output for human-readable stdout."""
    lines = [
        f"total_articles:    {stats['total_articles']}",
        f"clean_fraction:    {stats['clean_fraction']:.3f}",
    ]
    rc = stats["range_counts"]
    assert isinstance(rc, dict)
    lines.append(
        "range_counts:      "
        f"min={rc['min']} mean={rc['mean']:.2f} median={rc['median']} max={rc['max']}"
    )
    rl = stats["range_lengths"]
    assert isinstance(rl, dict)
    lines.append(
        "range_lengths(cp): "
        f"min={rl['min']} mean={rl['mean']:.2f} median={rl['median']} max={rl['max']}"
    )
    return "\n".join(lines)


@hydra.main(config_path=_CONFIG_PATH, config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    articles = load_oxen_tree(
        cfg.data.oxen_dir,
        stage=cfg.data.stage,
        require_nfc=cfg.data.require_nfc,
        range_units=cfg.data.range_units,
    )
    stats = compute_stats(articles)
    sys.stdout.write(render_stats(stats) + "\n")


if __name__ == "__main__":
    main()
