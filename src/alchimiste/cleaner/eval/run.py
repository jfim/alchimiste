"""`just eval` entrypoint — re-evaluate an existing artifact (TASK-022, REQ-009, NFR-006).

Loads the model + tokenizer + threshold from `<artifact_dir>/model/`,
loads articles from the same oxen tree the artifact was trained on
(unless an override is passed), restricts to the test split (per
`splits.json` inside the artifact), and re-runs `finalize` to produce
deterministic metrics.

NFR-006: two consecutive invocations yield byte-identical
metrics.json.
"""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from alchimiste.cleaner.data.loader import load_oxen_tree
from alchimiste.cleaner.data.split import SplitsManifest
from alchimiste.cleaner.training.finalize import finalize

_CONFIG_PATH = str(Path(__file__).resolve().parents[4] / "configs")


def evaluate(artifact_dir: Path, *, override_oxen_dir: Path | None = None) -> dict[str, object]:
    """Re-run the test-set evaluation against an existing artifact.

    `artifact_dir` must contain:
      * `config.yaml` — the resolved training config
      * `splits.json` — the partition manifest used during training
      * `model/`     — saved weights + tokenizer
    """
    artifact_dir = Path(artifact_dir)
    cfg = OmegaConf.load(artifact_dir / "config.yaml")
    splits = SplitsManifest.read_json(artifact_dir / "splits.json")

    oxen_dir = override_oxen_dir if override_oxen_dir is not None else Path(cfg.data.oxen_dir)
    articles = load_oxen_tree(
        oxen_dir,
        stage=cfg.data.stage,
        require_nfc=cfg.data.require_nfc,
        range_units=cfg.data.range_units,
        min_bytes=int(cfg.data.get("min_bytes", 0)),
    )
    by_id = {a.item_id: a for a in articles}
    val_arts = [by_id[i] for i in splits.val if i in by_id]
    test_arts = [by_id[i] for i in splits.test if i in by_id]

    # Load the tagger module dynamically — same lookup as training/loop.
    tagger_module = importlib.import_module(cfg.model.module)
    tagger_cls = getattr(tagger_module, "MODEL_CLASS", None) or tagger_module.Tagger
    tagger = tagger_cls.load(artifact_dir / "model")

    val_ex = tagger.tokenize(val_arts, max_seq_len=cfg.data.max_seq_len)
    test_ex = tagger.tokenize(test_arts, max_seq_len=cfg.data.max_seq_len)

    return finalize(
        tagger=tagger,
        val_articles=val_arts,
        val_examples=val_ex,
        test_articles=test_arts,
        test_examples=test_ex,
        artifact_dir=artifact_dir,
        eval_cfg=cfg.eval,
    )


@hydra.main(config_path=_CONFIG_PATH, config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    artifact = cfg.get("artifact")
    if not artifact:
        sys.exit(
            "just eval requires `artifact=<path>` (path to a previously-trained "
            "artifact directory). mlflow-run-id fetch is a planned follow-up."
        )
    override = cfg.get("override_oxen_dir")
    metrics = evaluate(
        Path(artifact),
        override_oxen_dir=Path(override) if override else None,
    )
    sys.stdout.write(
        json.dumps({k: v for k, v in metrics.items()}, sort_keys=True, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
