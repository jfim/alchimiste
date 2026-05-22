"""mlflow pyfunc wrapper for the text cleaner (TASK-024, REQ-011, REQ-012).

`CleanerModel.predict` is the single source of truth for the
text-in -> drop_ranges-out path. The stdin/stdout CLI (TASK-026)
shares this code so the two never drift.

Inference contract (REQ-012, REQ-013):
  * input: a UTF-8 markdown string, or a list of such strings.
  * the input is defensively NFC-normalized before tokenization
    (callers shouldn't have to know about the contract).
  * output: `{"drop_ranges": [[start, stop], ...]}` per input string,
    where ranges are **codepoint** offsets into the **NFC-normalized**
    input. A single-string input returns a single dict; a list input
    returns a list of dicts.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import mlflow.pyfunc

from alchimiste.cleaner.data.align import tokenize_and_align
from alchimiste.cleaner.data.loader import LabeledArticle
from alchimiste.cleaner.data.normalize import defensive_nfc
from alchimiste.cleaner.inference.decode import decode_token_runs

if TYPE_CHECKING:
    pass


class CleanerModel(mlflow.pyfunc.PythonModel):
    """The pyfunc shape mlflow registers and `mlflow.pyfunc.load_model` returns."""

    def load_context(self, context: mlflow.pyfunc.PythonModelContext) -> None:
        artifacts = context.artifacts
        self._artifact_dir = Path(artifacts["artifact_root"])
        threshold_path = self._artifact_dir / "threshold.json"
        threshold_payload = json.loads(threshold_path.read_text(encoding="utf-8"))
        self._threshold = float(threshold_payload["threshold"])
        self._max_seq_len = int(threshold_payload.get("max_seq_len", 512))

        # Resolve the tagger module the same way training/loop and
        # eval/run do — fully dynamic so the artifact stays
        # architecture-agnostic.
        from omegaconf import OmegaConf

        cfg = OmegaConf.load(self._artifact_dir / "config.yaml")
        import importlib

        module = importlib.import_module(cfg.model.module)
        cls = getattr(module, "MODEL_CLASS", None) or module.Tagger
        self._tagger = cls.load(self._artifact_dir / "model")
        # Override max_seq_len from cfg.data if available.
        self._max_seq_len = int(cfg.data.max_seq_len)

    def predict(
        self,
        context: mlflow.pyfunc.PythonModelContext,
        model_input,
        params: dict | None = None,
    ):
        """Either a single string or a list[str] -> drop_ranges per input."""
        single = isinstance(model_input, str)
        inputs: list[str] = [model_input] if single else list(model_input)
        results = [self._predict_one(text) for text in inputs]
        return results[0] if single else results

    # ------------------------------------------------------------------ #
    # Shared with the CLI                                                #
    # ------------------------------------------------------------------ #

    def _predict_one(self, text: str) -> dict:
        """One markdown string -> {"drop_ranges": [...]}."""
        normalized = defensive_nfc(text)
        article = LabeledArticle(
            item_id="_inference_",
            content_sha256="0" * 64,
            markdown_text=normalized,
            discard_ranges=(),
        )
        ex = tokenize_and_align(article, self._tagger.tokenizer, max_seq_len=self._max_seq_len)
        [probs] = self._tagger.predict_token_probs([ex])
        ranges = decode_token_runs(
            probs,
            ex.codepoint_offset_mapping,
            threshold=self._threshold,
        )
        return {"drop_ranges": [list(r) for r in ranges]}


def predict_text(artifact_dir: Path, text: str) -> dict:
    """Convenience: load an artifact and run a one-shot prediction.

    Used by the CLI (TASK-026) and by tests. Goes through the same
    `load_context` -> `_predict_one` pipeline as the mlflow pyfunc so
    the two paths are guaranteed identical.
    """
    model = CleanerModel()
    ctx = _LocalContext(artifact_dir=artifact_dir)
    model.load_context(ctx)  # type: ignore[arg-type]
    return model._predict_one(text)


class _LocalContext:
    """Minimal stand-in for `mlflow.pyfunc.PythonModelContext` so
    `predict_text` can call `load_context` without spinning up an mlflow
    run. `artifacts` is the only attribute the loader reads."""

    def __init__(self, artifact_dir: Path) -> None:
        self.artifacts = {"artifact_root": str(artifact_dir)}
