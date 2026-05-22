"""End-of-training finalization: threshold sweep, test eval, persist (TASK-020/021).

Split out of `training/loop.py` so the same logic can be re-invoked by
`just eval` (TASK-022) against a previously-trained artifact.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING

from alchimiste.cleaner.data.align import TokenizedExample
from alchimiste.cleaner.data.loader import LabeledArticle
from alchimiste.cleaner.eval.failures import ArticleResult, write_failures
from alchimiste.cleaner.eval.iou import aggregate_iou_metrics
from alchimiste.cleaner.eval.req003 import (
    clean_articles_kept_clean_fraction,
    excessive_ranges_fraction,
)
from alchimiste.cleaner.eval.threshold import ThresholdSelection, sweep_thresholds
from alchimiste.cleaner.inference.decode import decode_token_runs

if TYPE_CHECKING:
    from alchimiste.cleaner.models.base import TokenTagger

Range = tuple[int, int]


def finalize(
    *,
    tagger: TokenTagger,
    val_articles: Sequence[LabeledArticle],
    val_examples: Sequence[TokenizedExample],
    test_articles: Sequence[LabeledArticle],
    test_examples: Sequence[TokenizedExample],
    artifact_dir: Path,
    eval_cfg,
) -> dict[str, object]:
    """Run threshold selection + test evaluation, write artifacts, return metrics.

    The returned dict is suitable for `mlflow_io.log_final(...)` and is
    also dumped to `<artifact_dir>/metrics.json`.
    """
    # 1. Validation predictions for threshold sweep.
    val_probs = tagger.predict_token_probs(list(val_examples))
    val_offsets = [list(e.codepoint_offset_mapping) for e in val_examples]
    val_truth = [list(a.discard_ranges) for a in val_articles]

    threshold_pick = sweep_thresholds(
        per_example_probs=val_probs,
        per_example_offsets=val_offsets,
        per_example_true_ranges=val_truth,
        sweep_min=float(eval_cfg.threshold_sweep.min),
        sweep_max=float(eval_cfg.threshold_sweep.max),
        sweep_step=float(eval_cfg.threshold_sweep.step),
        iou_metric=float(eval_cfg.primary_iou),
        precision_floor=float(eval_cfg.precision_floor),
    )
    _write_threshold_json(threshold_pick, artifact_dir / "threshold.json")

    # 2. Test predictions at the chosen τ.
    test_probs = tagger.predict_token_probs(list(test_examples))
    test_pred = [
        decode_token_runs(probs, e.codepoint_offset_mapping, threshold=threshold_pick.threshold)
        for probs, e in zip(test_probs, test_examples, strict=True)
    ]
    test_truth = [list(a.discard_ranges) for a in test_articles]

    # 3. IoU metrics at every requested threshold (REQ-009).
    iou_thresholds = tuple(float(t) for t in eval_cfg.report_iou)
    iou_results = aggregate_iou_metrics(test_truth, test_pred, thresholds=iou_thresholds)

    # 4. REQ-003 sub-evaluations.
    clean_kept = clean_articles_kept_clean_fraction(test_truth, test_pred)
    excessive = excessive_ranges_fraction(
        test_pred,
        excessive_ranges_threshold=int(eval_cfg.excessive_ranges_threshold),
    )

    # 5. Persist failures.jsonl (REQ-010).
    failure_results = [
        ArticleResult(
            item_id=a.item_id,
            content_sha256=a.content_sha256,
            text=a.markdown_text,
            true_drop_ranges=tuple(a.discard_ranges),
            pred_drop_ranges=tuple(pred),
        )
        for a, pred in zip(test_articles, test_pred, strict=True)
    ]
    n_failures = write_failures(failure_results, artifact_dir / "failures.jsonl")

    # 6. Assemble the metric dict for mlflow + metrics.json.
    metrics: dict[str, object] = {
        "test/clean_articles_kept_clean_fraction": clean_kept,
        "test/excessive_ranges_fraction": excessive,
        "test/n_failures": float(n_failures),
        "val/selected_threshold": threshold_pick.threshold,
        "val/selected_threshold_precision": threshold_pick.precision_at_chosen,
        "val/selected_threshold_recall": threshold_pick.recall_at_chosen,
    }
    for tau, m in iou_results.items():
        # Use "p<tau>" style keys to keep mlflow happy (no @, no dots in
        # the integer prefix). Slash separates namespace from metric.
        tau_label = f"{tau:.2f}".rstrip("0").rstrip(".")
        metrics[f"test/iou_precision_at_{tau_label}"] = m.precision
        metrics[f"test/iou_recall_at_{tau_label}"] = m.recall
        metrics[f"test/iou_f1_at_{tau_label}"] = m.f1

    (artifact_dir / "metrics.json").write_text(
        json.dumps(_jsonable(metrics), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return metrics


def _write_threshold_json(pick: ThresholdSelection, dst: Path) -> None:
    payload = {
        "threshold": pick.threshold,
        "iou_metric": pick.iou_metric,
        "precision_at_chosen": pick.precision_at_chosen,
        "recall_at_chosen": pick.recall_at_chosen,
        "f1_at_chosen": pick.f1_at_chosen,
        "fell_back_to_max_precision": pick.fell_back_to_max_precision,
        "sweep": [
            {"threshold": t, "precision": p, "recall": r, "f1": f} for (t, p, r, f) in pick.sweep
        ],
    }
    dst.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _jsonable(d: dict[str, object]) -> dict[str, object]:
    """Convert dataclasses / sets / numpy floats to JSON-safe types."""
    out: dict[str, object] = {}
    for k, v in d.items():
        if hasattr(v, "__dataclass_fields__"):
            out[k] = asdict(v)
        else:
            out[k] = v
    return out
