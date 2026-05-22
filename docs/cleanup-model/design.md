# Design — Text Cleaner Model Training

**Project:** alchimiste
**Component:** Text cleaner model
**Version:** v1
**Status:** Draft

## 1. High-level shape

The text cleaner training pipeline is an offline batch process that consumes a versioned, content-addressed dataset (committed by `alchimiste pull cleaning` into an oxen working tree) and produces an mlflow-registered model. Inference is a thin path that loads the registered model and exposes a `predict` function plus a stdin/stdout CLI.

```
oxen working tree  →  loader  →  tokenize+align  →  train (GPU ok)  →  eval (IoU)  →  mlflow model
  cleaning/                                                                            (weights +
   rows.parquet                                                                         tokenizer +
   blobs/<sha>                                                                          python infer
                                                                                        + threshold)
   (REQ-001/002)      (REQ-002)    (REQ-004)         (REQ-005/006/007)  (REQ-009)      (REQ-011)
                                                                                              │
                                                                                              ▼
inference caller  ←  mlflow.pyfunc.load_model  →  predict(markdown) → {"drop_ranges": [...]}
                                                                                      (REQ-012)
```

Six separated concerns so model families can be experimented on in parallel (REQ-006) without perturbing the rest of the pipeline:

1. **`configs/`** — Hydra config tree; one composable config per model family.
2. **`data/`** — oxen-tree loader (with commit-identity detection), NFC normalization, codepoint↔token alignment, dataset splitting.
3. **`models/`** — pluggable architectures behind a common `TokenTagger` interface, one module per architecture.
4. **`training/`** — generic training loop driven by the Hydra config; threshold selection on val; mlflow logging.
5. **`eval/`** — IoU-based metrics + failure dumps re-ingestible by alambic.
6. **`inference/`** — the mlflow `pyfunc` wrapper, the CLI, and the codepoint-range decoder.

## 2. Data model

### 2.1 On-disk: oxen working tree (the dataset-puller's output)

```
<oxen-dir>/
  cleaning/
    rows.parquet          # schema below
    blobs/
      <sha256>            # raw UTF-8 bytes of NFC-normalized markdown; name == sha256 of the bytes
      ...
```

`rows.parquet` schema (matches the alambic dataset-export contract — REQ-002, REQ-013):

| Column                | Polars dtype          | Notes                                                                            |
| --------------------- | --------------------- | -------------------------------------------------------------------------------- |
| `item_id`             | `Utf8`                | Stable; used as partition key (REQ-008).                                         |
| `content_sha256`      | `Utf8` (64 hex)       | Joins to `cleaning/blobs/<sha>`. Hash is over the raw UTF-8 bytes of NFC text.   |
| `discard_ranges`      | `List(List(Int32))`   | Zero or more `[start, stop]` half-open **codepoint** offsets into the NFC text.  |
| `confirmed_at`        | `Int64` (unix s)      |                                                                                  |
| `updated_at`          | `Int64` (unix s)      |                                                                                  |
| `prior_model_version` | `Utf8` nullable       | Pre-label suggestion, if any. Not used as a training feature in v1.              |

Range invariants (validated by the loader, REQ-002): sorted ascending by `start`, non-overlapping, `0 ≤ start < stop ≤ codepoint_len(blob_text)`.

> **Cross-repo coordination note.** The alambic export plan as written today emits **byte** offsets (Elixir `binary_part` semantics). REQ-013 changes the contract to codepoint offsets over NFC text. Until alambic adopts this, the alchimiste loader can run in a transitional mode that converts byte offsets → codepoint offsets on read (using UTF-8 indexing on the NFC blob) — but the on-the-wire schema target is codepoint offsets. Tracked as Open Question Q3.

### 2.2 Dataset identity (oxen commit detection)

The oxen working tree alone does not pin the dataset version — the same tree can be at any commit, or dirty (uncommitted changes). The loader resolves this at training start:

1. Subprocess `oxen log -n 1 --short` (or equivalent) in `<oxen-dir>`, parse the HEAD commit hash.
2. Subprocess `oxen status` to detect uncommitted changes.
3. **If the tree is clean:** record `alchimiste.dataset.oxen_commit = <hash>` and `alchimiste.dataset.dirty = false` as mlflow run tags.
4. **If the tree is dirty:** record the commit hash anyway, set `dirty = true`, *and* require an explicit override flag (`--allow-dirty-data` or config `data.allow_dirty=true`) to proceed. Default behavior is to refuse, because a dirty tree means the recorded hash does not represent what was actually trained on.

This pattern matches the dataset-puller's own subprocess-based oxen wrapper, so we don't introduce a Python oxen dependency just for two read-only invocations.

### 2.3 Text encoding contract — codepoints over NFC (REQ-013)

`discard_ranges` are **codepoint offsets into NFC-normalized text**. Codepoints (rather than bytes or grapheme clusters) are picked because they are:

- well-defined and language-independent (Python `str` indexing is codepoint-based);
- stable across the UTF-8/UTF-16 representation boundary, so the model framework's choice of internal string encoding doesn't matter;
- the natural unit for most tokenizers, which align on codepoint spans, not byte spans.

The contract has three obligations:

- **alambic** stores blob text in NFC. The labeler operates on codepoint offsets into that NFC text, and the parquet export emits codepoint offsets directly (target state; see the Q3 transitional path).
- **alchimiste's loader** (`data/normalize.py`) decodes each blob from UTF-8 and asserts the result is byte-identical to its NFC re-normalization (`unicodedata.normalize("NFC", text) == text`). If `data.require_nfc` is true (default) and the check fails, the loader raises. If `data.require_nfc` is false, it normalizes defensively and emits a warning that the labels may be slightly misaligned.
- **Inference** (`inference/pyfunc.py`) defensively NFC-normalizes its input before tokenization. The returned `drop_ranges` are codepoint offsets into the **normalized** form. Callers needing offsets into the original (pre-normalization) input are responsible for the remap; v1 does not provide that remapper.

Tokens map back to codepoint offsets via § 2.5; alignment never traffics in bytes once past the loader.

### 2.4 In-memory: labeled example

```python
@dataclass(frozen=True)
class LabeledArticle:
    item_id: str
    content_sha256: str
    markdown_text: str                            # NFC-normalized text (decoded from UTF-8)
    discard_ranges: tuple[tuple[int, int], ...]   # codepoint offsets; sorted, non-overlapping
```

The blob is held as `str`, since labels are codepoint offsets and Python string indexing is codepoint-based. The original UTF-8 bytes are no longer needed past the loader; the content hash (`content_sha256`) is the audit trail back to the on-disk blob.

### 2.5 Tokenized example (post-alignment, REQ-004)

```python
@dataclass(frozen=True)
class TokenizedExample:
    item_id: str
    input_ids: list[int]
    codepoint_offset_mapping: list[tuple[int, int]]   # codepoint spans per token, into the NFC text
    labels: list[int]                                 # 0 keep, 1 drop, -100 special
```

`codepoint_offset_mapping` stores `[start_codepoint, end_codepoint)` of each token in the NFC text. Most HuggingFace fast tokenizers report char (= codepoint, for non-surrogate Python strings) offsets directly; for tokenizers that report byte offsets, the loader converts via UTF-8 indexing on the NFC text.

A token is labeled `drop` iff its codepoint interval is **entirely contained** in some `discard_range`. Tokens straddling a boundary are labeled `keep` (NFR-002 precision bias).

### 2.6 Partition manifest

`splits.json` lists `item_id`s per partition. Partitioning is by **hash on `item_id`** with a fixed seed (REQ-008): newly-pulled rows are routed deterministically into partitions on each retrain, so an item never silently migrates between train and test as the corpus grows under the active-loop (NFR-007). The manifest is regenerated each run from the current corpus and the seed; assignment is stable because the hash is.

## 3. Module layout

```
src/alchimiste/
  cleaner/
    __init__.py
    data/
      __init__.py
      loader.py            # REQ-001/002: read oxen tree, join rows + blobs
      oxen_meta.py         # § 2.2: subprocess `oxen log` / `oxen status`
      align.py             # REQ-004: codepoint↔token alignment
      split.py             # REQ-008: hash-on-item_id splitter
      normalize.py         # § 2.3: NFC assertion + defensive normalization
    models/
      __init__.py
      base.py              # REQ-006: TokenTagger Protocol
      # one module per architecture (parallel experiments — § 5.1):
      #   encoder_hf.py, crf_on_embeddings.py, lora_seqtagger.py, ...
    training/
      __init__.py
      loop.py              # REQ-005: train + select threshold; logs to mlflow
      mlflow_io.py         # § 5.3: live + final mlflow logging helpers
    eval/
      __init__.py
      iou.py               # REQ-009: IoU-based metrics (primary)
      diagnostics.py       # token-level metrics (diagnostic only)
      failures.py          # REQ-010: alambic-re-ingestible failures.jsonl
    inference/
      __init__.py
      pyfunc.py            # REQ-011/012: mlflow pyfunc wrapper
      decode.py            # token-runs → codepoint ranges
      cli.py               # REQ-012: stdin → JSON stdout

configs/                   # Hydra config tree (§ 6)
  config.yaml              # default composition
  data/
    default.yaml
  model/
    encoder_hf.yaml        # one yaml per architecture
    crf_on_embeddings.yaml
    lora_seqtagger.yaml
  training/
    default.yaml
  eval/
    default.yaml

tests/
  cleaner/
    fixtures/
      oxen_tree/           # tiny synthesized oxen working tree
    test_loader.py
    test_oxen_meta.py
    test_normalize.py
    test_align.py
    test_split.py
    test_iou.py
    test_pyfunc.py
```

## 4. Interfaces

### 4.1 Model interface (REQ-006)

```python
class TokenTagger(Protocol):
    def fit(
        self,
        train: list[TokenizedExample],
        val: list[TokenizedExample],
        cfg: DictConfig,                        # Hydra config sub-tree for this model
        callbacks: TrainingCallbacks,           # batch/epoch hooks for mlflow logging
    ) -> None: ...
    def predict_token_probs(self, examples: list[TokenizedExample]) -> list[list[float]]: ...
    def save(self, dst: Path) -> None: ...      # weights + tokenizer config
    @classmethod
    def load(cls, src: Path) -> "TokenTagger": ...
```

`predict_token_probs` returns per-token drop-class probabilities so threshold selection (§ 5.2) lives in one place. `TrainingCallbacks` exposes `on_batch_end(step, loss)` and `on_epoch_end(epoch, val_metrics)` hooks — implementations are required to call them, which is how live mlflow metric logging works regardless of framework.

### 4.2 Inference contract (REQ-012)

The model is published as an mlflow `python_function` whose `predict` accepts either a single string or a list of strings:

```python
class CleanerModel(mlflow.pyfunc.PythonModel):
    def load_context(self, context): ...
    def predict(self, context, model_input: list[str] | str) -> list[dict] | dict:
        # Normalizes input to NFC (§ 2.3), then returns
        # {"drop_ranges": [[start, stop], ...]} per input markdown string
```

The CLI (`alchimiste-clean` or `python -m alchimiste.cleaner.inference.cli`) reads UTF-8 markdown from stdin and writes one JSON object to stdout — calling the same code path as the pyfunc.

Outputs are sorted, non-overlapping, half-open **codepoint** ranges into the NFC-normalized input.

### 4.3 Codepoint-range decoding (token-runs → ranges)

1. Threshold per-token drop probabilities (threshold loaded from artifact).
2. Group consecutive `drop` tokens into runs.
3. For each run, emit `[min(codepoint_offset_start), max(codepoint_offset_end))`.
4. Apply a configurable minimum-run-length filter (default 0).
5. Sort and merge any adjacent/overlapping ranges produced by sub-word tokenizers.

## 5. Training, threshold selection, and parallel experiments

### 5.1 Parallel experiments

Each architecture is a self-contained module under `models/` plus a self-contained yaml under `configs/model/`. Running two experiments in parallel is two independent invocations:

```sh
just train model=encoder_hf      # terminal A
just train model=crf_on_embeddings   # terminal B
```

Or via Hydra multirun for sweeps:

```sh
uv run python -m alchimiste.cleaner.training.loop -m model=encoder_hf,crf_on_embeddings training.seed=11,17,42
```

Each run gets:
- A fresh Hydra `outputs/<timestamp>/` working directory (no shared mutable paths).
- An independent mlflow run id under the same experiment.
- The same registered model name (`alchimiste-text-cleaner`) — registry promotion uses run metrics to pick a winner; competing runs never overwrite each other's artifacts.

No code path writes to `runs/latest/`, `artifact/current/`, or any other shared mutable location.

### 5.2 Loss and imbalance handling (REQ-007)

Default loss is per-token weighted cross-entropy with a configurable `training.class_weight_drop` (default 5.0 — to be revisited after the first end-to-end run). Implementations may substitute another loss as long as `predict_token_probs` outputs are usable for threshold sweeping (see Q5).

### 5.3 Threshold selection (NFR-002, REQ-007)

After training, on the **validation** split:

1. Run `predict_token_probs`.
2. Decode at thresholds τ ∈ {0.30, 0.35, …, 0.90}.
3. Compute IoU-based drop-range precision/recall at IoU = 0.5 for each τ.
4. Pick the **lowest τ that meets `eval.precision_floor` (default 0.85)** — maximizes recall while honoring the precision bias.
5. If no τ meets the floor, pick the τ with maximum precision and tag the mlflow run `alchimiste.threshold.fell_back_to_max_precision = true`.
6. Persist the chosen τ inside the mlflow artifact (REQ-011) so inference uses the same value.

### 5.4 mlflow integration (REQ-011, NFR-007, and REQ-009 metric publishing)

Every training run logs to mlflow:

**Live (during training):**
- `train/loss` at every batch step via `on_batch_end`.
- `train/epoch_loss`, `val/iou_f1@0.5`, `val/iou_precision@0.5`, `val/iou_recall@0.5`, `val/token_f1` at every epoch via `on_epoch_end`.
- `val/threshold_sweep/<τ>/precision`, `.../recall`, `.../f1` after each epoch (so threshold drift can be watched live).

**At run end:**
- All keys from `metrics.json` (test-set IoU at τ ∈ {0.5, 0.9, 1.0}, `clean_articles_kept_clean_fraction`, `excessive_ranges_fraction`).
- `failures.jsonl` and `splits.json` as run artifacts.
- The full `python_function` model as the run's single mlflow model.

**Tags:**
- `alchimiste.dataset.oxen_commit` — from § 2.2.
- `alchimiste.dataset.dirty` — boolean.
- `alchimiste.dataset.oxen_dir` — for human readability.
- `alchimiste.model.architecture` — mirror of `cfg.model.name` for filtering in the mlflow UI.
- `alchimiste.threshold.value` and `alchimiste.threshold.iou` — the selected operating point.

Registered under model name `alchimiste-text-cleaner` (configurable via `mlflow.model_name` in the config).

## 6. Configuration — Hydra

The pipeline is configured by Hydra. Top-level `configs/config.yaml`:

```yaml
# configs/config.yaml
defaults:
  - data: default
  - model: encoder_hf        # default architecture; override on the CLI
  - training: default
  - eval: default
  - _self_

mlflow:
  experiment: alchimiste-text-cleaner
  model_name: alchimiste-text-cleaner
  tracking_uri: ${oc.env:MLFLOW_TRACKING_URI, file://./mlruns}

seed: 17                     # reproducibility (NFR-001); also seeds split.py
```

`configs/data/default.yaml`:

```yaml
oxen_dir: ./alchimiste-data
stage: cleaning
allow_dirty: false           # § 2.2
require_nfc: true            # § 2.3
max_seq_len: 512
```

`configs/training/default.yaml`:

```yaml
batch_size: 8
learning_rate: 3e-5
epochs: 6
class_weight_drop: 5.0       # REQ-007
device: auto                 # cuda if available, else cpu (training only; inference is CPU)
```

`configs/eval/default.yaml`:

```yaml
primary_iou: 0.5             # REQ-009
report_iou: [0.5, 0.9, 1.0]
precision_floor: 0.85        # NFR-002
threshold_sweep:
  min: 0.30
  max: 0.90
  step: 0.05
excessive_ranges_threshold: 6  # REQ-003 smell flag
```

`configs/model/encoder_hf.yaml` (one example):

```yaml
# @package model
name: encoder_hf
module: alchimiste.cleaner.models.encoder_hf
hf_model_name: distilbert-base-uncased
```

Each model yaml specifies a `module` path and any architecture-specific parameters. The training loop dynamically imports `model.module` and instantiates its `TokenTagger`. Adding a new architecture is: one new module under `src/alchimiste/cleaner/models/`, one new yaml under `configs/model/`. No edits to anything else (REQ-006).

## 7. Evaluation

### 7.1 Primary metric — IoU over codepoint ranges (REQ-009)

For each test article:
1. Predict drop ranges.
2. For each predicted range, find the ground-truth range with max IoU.
3. At τ ∈ {0.5, 0.9, 1.0}, count TPs (best-IoU ≥ τ), FPs, FNs.

Report precision, recall, F1 at each τ, plus micro-averaged totals. Logged to mlflow as `test/iou_{precision,recall,f1}@<τ>` and mirrored to a local `metrics.json` for offline inspection.

### 7.2 Diagnostic metrics

Token-level precision/recall/F1 of the drop class is logged but not gated on. Useful for debugging tokenizer/alignment issues.

### 7.3 Clean-only sub-eval (REQ-003)

Two derived numbers logged as `test/clean_articles_kept_clean_fraction` and `test/excessive_ranges_fraction` (using `eval.excessive_ranges_threshold` from the config). These flag for human review rather than fail-gating.

### 7.4 Failure dump (REQ-010)

`failures.jsonl` is written next to `metrics.json` and also logged as an mlflow artifact:

```json
{
  "item_id": "...",
  "content_sha256": "...",
  "true_drop_ranges": [[s, e], ...],
  "pred_drop_ranges": [[s, e], ...],
  "false_positive_ranges": [[s, e], ...],
  "false_negative_ranges": [[s, e], ...],
  "kept_diff": "unified-diff-style rendering"
}
```

Schema is a superset of what alambic needs to render a review queue — exact ingest hook tracked as Open Question Q4.

## 8. Artifact contents (REQ-011)

The mlflow model is a `python_function` flavor with this layout under `artifacts/`:

```
artifacts/
  model/                  # framework-native save (weights + tokenizer)
  config.yaml             # resolved Hydra config (full, including overrides)
  threshold.json          # {"threshold": 0.6, "iou_metric": 0.5, ...}
  splits.json             # partition manifest snapshot
  metrics.json            # final test metrics (also in mlflow)
  failures.jsonl          # per-article failure dump (also in mlflow)
  inference/              # the Python code needed at predict time
    pyfunc.py             # CleanerModel class
    decode.py             # codepoint-range decoder
    align.py              # tokenizer side of codepoint↔token alignment
    normalize.py          # NFC normalization (§ 2.3)
```

The `inference/` subdirectory ships **with** the model so an inference-time process only needs the mlflow client plus the model framework's pip dependencies. `requirements.txt` is generated from the active uv environment and included in the mlflow model.

## 9. Tooling (NFR-005)

```
just train [model=encoder_hf] [overrides...]
                  # full run; registers an mlflow model. Forwards extra args to Hydra.
just train-multi model=encoder_hf,crf_on_embeddings seed=11,17,42
                  # Hydra -m multirun convenience.
just eval run=<mlflow-run-id> [data.oxen_dir=...]
just predict run=<mlflow-run-id>            # stdin → JSON stdout
just label-stats [data.oxen_dir=...]        # corpus statistics
```

All recipes are thin wrappers around `uv run python -m alchimiste.cleaner.<entrypoint> <hydra-args>`.

## 10. Requirement → Design mapping

| REQ / NFR | Design element                                                                            |
| --------- | ----------------------------------------------------------------------------------------- |
| REQ-001   | `data/loader.py` reads `<oxen-dir>/cleaning/{rows.parquet, blobs/}`                       |
| REQ-002   | § 2.1 schema; loader joins rows to blobs; range invariants enforced                       |
| REQ-003   | `test/clean_articles_kept_clean_fraction` + `test/excessive_ranges_fraction` (§ 7.3)      |
| REQ-004   | `data/align.py` + `TokenizedExample.codepoint_offset_mapping` + `inference/decode.py` (§ 4.3) |
| REQ-013   | `data/normalize.py` NFC assertion (§ 2.3) + `inference/pyfunc.py` defensive normalize     |
| REQ-014   | `data/oxen_meta.py` subprocess (§ 2.2) + mlflow tags (§ 5.4)                              |
| REQ-015   | `training/mlflow_io.py` callbacks; live + end-of-run logging (§ 5.4)                      |
| REQ-005   | `training/loop.py` driven by Hydra config; model module dynamically imported              |
| REQ-006   | `models/base.py` `TokenTagger` Protocol; one module + one yaml per architecture (§ 5.1)   |
| REQ-007   | `training.class_weight_drop` + threshold selection (§ 5.3)                                |
| REQ-008   | `data/split.py` — hash-on-`item_id` with `seed` from config (§ 2.6)                       |
| REQ-009   | `eval/iou.py` (primary, logged live + at end); `eval/diagnostics.py` (token, diagnostic)  |
| REQ-010   | `eval/failures.py` writes alambic-re-ingestible `failures.jsonl` (§ 7.4)                  |
| REQ-011   | mlflow pyfunc registration (§ 5.4) + § 8 artifact layout                                  |
| REQ-012   | `inference/pyfunc.py` + `inference/cli.py` sharing the same code path                     |
| NFR-001   | `seed` plumbed through Hydra; deterministic dataloader; oxen commit hash logged           |
| NFR-002   | `eval.precision_floor` + conservative boundary-token labeling                             |
| NFR-003   | `training.device: auto` for GPU; inference pyfunc is CPU-only                             |
| NFR-004   | mlflow handles distribution; soft warning at 2 GB inside training loop                    |
| NFR-005   | `justfile` recipes wrapping `uv run` + Hydra                                              |
| NFR-006   | Greedy decoding only; seed pinned in eval                                                 |
| NFR-007   | mlflow run tags `alchimiste.dataset.oxen_commit` + `.dirty`; single-command re-run        |
| NFR-008   | Parallel experiments via independent Hydra outputs + mlflow runs (§ 5.1); no shared paths |
| NFR-009   | Hydra config tree (§ 6); every parameter overridable from the CLI                         |

## 11. Open Questions

1. **Q1 — Architecture pick(s) for the first runs.** Plausible candidates: HF encoder + token-classification head (DistilBERT / MiniLM / ModernBERT), small encoder + CRF, small LoRA seq-tagger. v1 ships at least one module under `models/`; parallel experimentation (§ 5.1) decides which gets promoted in the registry first.
2. **Q2 — Long-document handling.** v1 truncates to `data.max_seq_len`. Open: should the loader *log* truncated articles so we can size them and decide if v2 needs chunked inference?
3. **Q3 — Codepoint-offset + NFC adoption in alambic.** REQ-013 / § 2.3 requires alambic to (a) store blob text in NFC and (b) emit `discard_ranges` as codepoint offsets into that NFC text. Alambic's current export emits byte offsets (Elixir `binary_part`-based). This is a cross-repo contract change. Until alambic adopts both halves, the alchimiste loader operates in a transitional mode: it NFC-normalizes on read (or asserts and fails, per `data.require_nfc`) and converts byte offsets → codepoint offsets via UTF-8 indexing on the normalized text. Drop the transitional conversion once alambic emits codepoint offsets natively.
4. **Q4 — Failure-dump ingest into alambic.** `failures.jsonl` is written in a re-ingestible shape, but alambic does not yet have an importer. Coordinate column names and an upload endpoint before v2.
5. **Q5 — Calibration assumption.** Threshold selection (§ 5.3) assumes `predict_token_probs` outputs are monotonic-with-confidence. Architectures with non-calibrated outputs (e.g., CRF) need a different threshold strategy — likely picking the operating point on a Viterbi-decoded label sequence rather than per-token probabilities.
6. **Q6 — Cham markdown stability.** Inference assumes the markdown supplied at predict time is identical (after NFC normalization) to what Cham produced at label time. If Cham re-extracts and produces different text (e.g., updated rules), the labeler's offsets are stale. The current architecture sidesteps this by keying labels on `content_sha256` and storing the labeled blob in alambic, but inference callers must use the same Cham version or accept drift.
