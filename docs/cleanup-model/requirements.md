# Requirements — Text Cleaner Model Training

**Project:** alchimiste
**Component:** Text cleaner model
**Version:** v1
**Status:** Draft

## Overview

The text cleaner is one of two ML models in alchimiste (the other being the DOM node extractor). Given the markdown produced by the upstream Cham extraction step for an article, the text cleaner identifies contiguous ranges of text that are *not* part of the primary article content (e.g. "Read the original article…", audio-player fallbacks, survey CTAs, repeated nav cruft) and emits those ranges so the downstream consumer can remove them. Range offsets are codepoint indices into the NFC-normalized form of the input (see REQ-013).

This document specifies the requirements for *training* that model — the dataset it consumes, the training pipeline, the evaluation gates, and the artifact it produces. It does not specify the labeling workflow (owned by `~/projects/alambic`) or the dataset transport (owned by the `alchimiste pull` CLI, see `docs/superpowers/plans/2026-05-21-dataset-puller.md`).

## Actors

- **Trainer** — the person (or CI / scheduled job) running the training pipeline.
- **Labeler** — a person using `~/projects/alambic` to confirm cleaning labels.
- **Inference caller** — the downstream Python process that loads the trained model and queries it on a new article's markdown.

## Functional Requirements

### Data

- **REQ-001** — The training pipeline shall consume the `cleaning` dataset produced by `alchimiste pull cleaning <oxen-dir>` and stored in an oxen-tracked working tree at `<oxen-dir>/cleaning/`.
  *Acceptance:* Given an oxen working tree containing `cleaning/rows.parquet` and `cleaning/blobs/<sha256>` for every referenced hash, the loader yields one in-memory example per row with no schema errors and no missing-blob errors.

- **REQ-002** — Each labeled example shall be assembled from one row of `cleaning/rows.parquet` plus the blob at `cleaning/blobs/<row.content_sha256>`. The row's schema is the export contract defined by the alambic dataset-export plan:

  | Column                | Type                  | Meaning                                                                                       |
  | --------------------- | --------------------- | --------------------------------------------------------------------------------------------- |
  | `item_id`             | utf8                  | Stable identifier; used as partition key.                                                     |
  | `content_sha256`      | utf8 (64 hex chars)   | sha256 of the raw blob bytes. Joins to `cleaning/blobs/<sha>`.                                |
  | `discard_ranges`      | list\[list\[int32\]\] | Zero or more `[start, stop]` half-open **codepoint** offsets into the NFC-normalized blob (REQ-013). Sorted, non-overlapping. |
  | `confirmed_at`        | int64 (unix s)        | When the labeler confirmed this row.                                                          |
  | `updated_at`          | int64 (unix s)        | Last modification.                                                                            |
  | `prior_model_version` | utf8 (nullable)       | Model version that produced the pre-labeling suggestion, if any.                              |

  *Acceptance:* For every fixture row, removing each `[start, stop)` codepoint range from the NFC-normalized form of the joined blob yields a "kept" string identical to the labeler's reference. Out-of-bounds, inverted (`stop <= start`), or overlapping ranges fail the loader with a clear error.
  *Note:* The alambic side currently exports byte offsets; aligning alambic to emit codepoint offsets over NFC-normalized text is a cross-repo coordination item tracked alongside REQ-013.

- **REQ-003** — The pipeline shall handle the common case where `discard_ranges == []` (nothing to clean) and shall not bias the trained model toward over-dropping.
  *Acceptance:* Two gates on the held-out test set —
  (a) on articles whose ground-truth `discard_ranges` is empty, the model predicts zero drop ranges on ≥ 95% of them;
  (b) the model produces ≤ 6 predicted drop ranges per article on ≥ 95% of articles. Articles exceeding either threshold are flagged for human review rather than treated as automatic failures (the threshold is a smell, not a hard error).

- **REQ-004** — The pipeline shall convert codepoint-level labels into the model's tokenization (token or natural-boundary), train against that, and convert the model's predictions back into codepoint ranges for output and evaluation.
  *Acceptance:* For each fixture, the codepoint→token→codepoint round-trip reproduces the original `discard_ranges` within ±1 codepoint at each boundary.

### Training

- **REQ-005** — The pipeline shall train a model that maps a markdown article (NFC-normalized text) to a set of `[start, stop)` drop ranges over its codepoints, evaluated under the gates in REQ-003, REQ-007, and REQ-009. The choice of model family, tokenizer, and loss is left to experimentation in design.md.
  *Acceptance:* `just train` produces an artifact that satisfies all evaluation gates on the held-out test set.

- **REQ-006** — The pipeline shall keep the model family pluggable: swapping in a second architecture shall not require changes to the data loader, evaluator, or inference adapter.
  *Acceptance:* A second architecture can be added by introducing one new module + one config entry; no edits to data, eval, or inference modules are needed.

- **REQ-007** — The pipeline shall counteract the heavy "keep"-class imbalance (most articles have empty `discard_ranges`) without collapsing to a trivial "drop nothing" classifier on the labeled minority.
  *Acceptance (initial target; may be revised after the first end-to-end run):* on the test set restricted to articles with at least one ground-truth drop range, drop-range **recall ≥ 0.70** at drop-range **precision ≥ 0.85** at IoU = 0.5 (see REQ-009 for metric definition).

### Evaluation

- **REQ-008** — The pipeline shall split the labeled corpus into train / validation / test partitions with a fixed reproducible seed, keyed on `item_id`, and shall never let an `item_id` appear in more than one partition.
  *Acceptance:* Running the splitter twice with the same seed yields byte-identical partition manifests.

- **REQ-009** — The primary evaluation metric shall be **IoU over codepoint ranges** — for each predicted range, find the best-IoU ground-truth range and count it as a true positive if IoU ≥ τ. The evaluator shall report precision / recall / F1 of the drop class at τ ∈ {0.5, 0.9}, plus exact-match (τ = 1.0). Token-level metrics are reported as a diagnostic only; IoU is the framing that matches the task ("text segmentation").
  *Acceptance:* `just eval` prints all three IoU thresholds and writes `metrics.json` with the same numbers.

- **REQ-010** — The evaluator shall surface per-article failure cases (false-positive and false-negative ranges) in a format that can be round-tripped back into `~/projects/alambic` for labeler review.
  *Acceptance:* A `failures.jsonl` is produced with one row per failing test article containing `item_id`, `content_sha256`, `true_drop_ranges`, `pred_drop_ranges`, and a diff. An open question (Q3 in design.md) tracks the exact alambic-side ingestion shape; v1 commits to the file existing in a re-ingestable JSONL form even if alambic doesn't consume it yet.

### Artifact & Inference Contract

- **REQ-011** — Training shall produce a single self-contained artifact comprising **(a) model weights, (b) tokenizer / preprocessing parameters, (c) the Python inference code needed to load and call the model, and (d) the decision threshold selected on the validation set.** The artifact shall be publishable to the project's mlflow model registry as a single mlflow model.
  *Acceptance:* A fresh Python process with only the mlflow client installed can `mlflow.pyfunc.load_model(...)` the registered artifact and produce predictions on a fixture article — no checkout of the alchimiste repo required at inference time.

- **REQ-012** — The mlflow-loaded model's `predict` shall accept one or more UTF-8 markdown strings (NFC-normalized; see REQ-013 for defensive normalization) and return, for each, a JSON-serializable object of the form `{"drop_ranges": [[start, stop], ...]}` where each pair is a half-open **codepoint** range into the NFC-normalized input. The same predict path shall also be invocable via a CLI that reads markdown on stdin and writes the JSON to stdout.
  *Acceptance:* On 5 fixture articles, both invocation paths produce identical, valid, sorted, non-overlapping output and match the model's internal predictions.

- **REQ-013** — The text-encoding contract for labels, inference inputs, and predicted ranges shall be **codepoint offsets into Unicode NFC-normalized text**, decoded from UTF-8. The pipeline shall:
  (a) assert on load that each dataset blob, when decoded as UTF-8 and re-normalized to NFC, is byte-identical to its stored form — and fail loudly otherwise;
  (b) defensively NFC-normalize inference inputs before tokenization, so that callers passing non-normalized text get correct results against the normalized form (the returned ranges refer to the normalized text, not the caller's original — callers needing a remap to the pre-normalization text are responsible for it).
  *Acceptance:* A fixture containing decomposed-form characters (e.g. NFD "é" as `e` + combining acute) fails the load assertion until normalized; once normalized, the same article round-trips through inference and produces ranges that, when applied to the NFC form, yield the expected kept text.
  *Cross-repo note:* This requires alambic to store and export labels keyed to the NFC form. Coordination is tracked in design.md Q3.

- **REQ-014** — Each training run shall record the exact identity of the dataset it consumed: the oxen commit hash of the working tree at training start, plus a clean/dirty flag from `oxen status`. By default the pipeline shall refuse to start training against a dirty working tree (the recorded commit would not represent what was trained on); an explicit opt-in flag shall be required to override.
  *Acceptance:* (a) Running `just train` against a clean working tree records `alchimiste.dataset.oxen_commit = <hash>` and `alchimiste.dataset.dirty = false` on the mlflow run. (b) Running against a dirty tree without the override flag exits non-zero with a clear error; with the override, the run starts and is tagged `dirty = true`.

- **REQ-015** — The training pipeline shall publish metrics to mlflow both *live during training* and *at completion*, not only as a local JSON file. Live metrics shall include per-batch training loss and per-epoch validation metrics (at minimum: epoch loss, IoU F1 / precision / recall at IoU=0.5, and the validation threshold sweep). End-of-run metrics shall include all test-set numbers from REQ-009 and the REQ-003 smell fractions.
  *Acceptance:* Opening the mlflow UI on a completed run shows non-empty time-series for `train/loss` and per-epoch `val/*` metrics, plus scalar end-of-run `test/*` metrics. The same numbers appear in `metrics.json` for offline inspection.

## Non-Functional Requirements

- **NFR-001 — Reproducibility.** A given (data snapshot via oxen commit, config, seed) triple shall produce test-set IoU-F1 within ±0.5 points across re-runs on the same machine.
- **NFR-002 — Precision bias.** A false positive (dropping real article text) is far more damaging than a false negative (leaving boilerplate in). The decision threshold shall be selected on the validation set to favor precision; the target floor is drop-range precision ≥ 0.85 at the chosen operating threshold (IoU = 0.5).
- **NFR-003 — Hardware split.** Training may use a GPU. Inference shall run on CPU only and shall complete in ≤ 300 seconds per article on a modern laptop CPU.
- **NFR-004 — Footprint.** No hard cap. The artifact is distributed via the mlflow model registry, so on-disk size is bounded by what mlflow can practically register and retrieve. A soft warning is emitted at training time if the artifact exceeds 2 GB so we notice runaway choices.
- **NFR-005 — Tooling consistency.** All training, evaluation, and inference entry points shall be invocable via `uv run` and surfaced as `just` recipes, matching the existing alchimiste conventions.
- **NFR-006 — Determinism in evaluation.** Evaluation runs shall be deterministic given a fixed artifact and test set.
- **NFR-007 — Active retraining.** The training pipeline is part of an active labeling loop: as the labeler adds or revises rows in alambic, a fresh oxen pull yields a new dataset commit and a re-training is expected. The pipeline shall make this re-run cheap to invoke (a single `just train` against the current oxen commit) and shall record the source oxen commit hash inside the mlflow run for provenance (see REQ-014).
- **NFR-008 — Parallel experiments.** Multiple training runs (e.g. different model architectures, hyperparameter sweeps, or seeds) shall be runnable concurrently on the same machine and against the same dataset commit without interfering with each other. No code path shall write to a shared mutable location (`runs/latest/`, `artifact/current/`, fixed-name temp files, etc.); each run shall be fully isolated to its own working directory and its own mlflow run id. Two simultaneous `just train` invocations differing only in architecture or seed shall both complete successfully and both register their artifacts under the same mlflow model name.
- **NFR-009 — Composable, override-friendly configuration.** All run-time parameters (model architecture, hyperparameters, dataset path, mlflow settings, eval thresholds) shall be expressed in a composable config tree rather than scattered across code or environment variables. Any single parameter shall be overridable from the command line for one-off experiments without editing config files on disk.

## Out of Scope (v1)

- DOM node extraction — handled by the sibling model in alchimiste.
- The labeling UI itself — owned by `~/projects/alambic`.
- The dataset transport — owned by `alchimiste pull` (dataset-puller plan).
- mlflow tracking-server provisioning — assumed already available; the training pipeline only writes to it.
- Multilingual support — v1 targets English-language articles only; behavior on other languages is undefined.
- Streaming or chunked inference for very long documents — v1 assumes the whole markdown fits in one forward pass; documents exceeding the model's context window are an out-of-scope failure mode for v1.
- Rewriting or paraphrasing text — the model only flags ranges to drop; it never generates replacement text.
- Auth on the dataset endpoints — same punt as the alambic side.
