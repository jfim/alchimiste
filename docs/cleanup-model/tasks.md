# Tasks — Text Cleaner Model Training

**Project:** alchimiste
**Component:** Text cleaner model
**Version:** v1
**Status:** Draft

Tasks are ordered: scaffolding → data layer → model + training loop → evaluation → mlflow + artifact → inference → end-to-end validation. Every task references the requirements / NFRs it satisfies inline on the checkbox line, and includes an explicit verification step. Tasks are sized to keep each under ~200 lines of new code.

---

## Phase 1 — Scaffolding

Goal: stand up the package tree, dependencies, Hydra config skeleton, and `just` recipes so subsequent phases have a place to land. Nothing model-specific yet.

- [ ] **TASK-001** [NFR-005, NFR-009] Create the `src/alchimiste/cleaner/` package tree (`data/`, `models/`, `training/`, `eval/`, `inference/`) with empty `__init__.py` files matching design.md § 3.
  *Verify:* `uv run python -c "import alchimiste.cleaner.data, alchimiste.cleaner.models, alchimiste.cleaner.training, alchimiste.cleaner.eval, alchimiste.cleaner.inference"` exits 0.

- [ ] **TASK-002** [REQ-005, REQ-011, REQ-015, NFR-005, NFR-009] Add runtime dependencies via `uv add`: `transformers`, `tokenizers`, `torch` (pinned to a CPU wheel — inference runs CPU-only per NFR-003), `numpy`, `scikit-learn`, `mlflow`, `hydra-core`, `omegaconf`. `polars` is already present from the dataset-puller work.
  *Verify:* `just sync` succeeds; `uv run python -c "import torch, transformers, mlflow, hydra, omegaconf"` exits 0; the resolved `torch` wheel has no CUDA dependency in `uv.lock`.

- [ ] **TASK-003** [NFR-009] Create the `configs/` tree (`config.yaml`, `data/default.yaml`, `training/default.yaml`, `eval/default.yaml`) with the contents from design.md § 6. `configs/model/` stays empty for now (filled by TASK-014).
  *Verify:* `uv run python -c "from hydra import compose, initialize; initialize(config_path='../configs', version_base=None); print(compose('config', overrides=['model=encoder_hf']))"` parses without errors once a placeholder `configs/model/encoder_hf.yaml` is in place (or use `+model={}` to bypass for the smoke check).

- [ ] **TASK-004** [NFR-005] Add empty `just` recipes for `train`, `train-multi`, `eval`, `predict`, and `label-stats` that each print a placeholder message and forward extra args to the eventual Hydra entrypoint. These get filled in by later tasks.
  *Verify:* `just train` and `just train-multi` print their placeholders and exit 0.

---

## Phase 2 — Data layer

Goal: turn an oxen working tree into in-memory `LabeledArticle` and `TokenizedExample` objects with codepoint-correct labels.

- [ ] **TASK-005** [REQ-014] Implement `data/oxen_meta.py::read_commit(oxen_dir) -> OxenMeta` with fields `commit_hash: str` and `dirty: bool`. Subprocess `oxen log -n 1` and `oxen status`; parse the commit hash and the clean/dirty signal from their output.
  *Verify:* Unit test using a tmp_path oxen repo (or a stubbed subprocess) confirms (a) clean tree yields `dirty=False` and a 40-char hash; (b) introducing an untracked file flips `dirty=True`; (c) missing `oxen` binary raises a clear error.

- [ ] **TASK-006** [REQ-013] Implement `data/normalize.py::assert_nfc(text: str) -> None` (raises on non-NFC) and `defensive_nfc(text: str) -> str` (always returns NFC, no error). Both are thin wrappers around `unicodedata.normalize("NFC", ...)`.
  *Verify:* Unit test with the canonical NFD-vs-NFC example (`"é"` vs `"é"`) confirms `assert_nfc` raises on the NFD form and passes on the NFC form; `defensive_nfc` round-trips both to the NFC form.

- [ ] **TASK-007** [REQ-001, REQ-002, REQ-013] Implement `data/loader.py::load_oxen_tree(oxen_dir, stage, require_nfc=True) -> list[LabeledArticle]`. Reads `<stage>/rows.parquet` via polars, joins each row to `<stage>/blobs/<content_sha256>`, decodes the blob as UTF-8, runs `assert_nfc` or `defensive_nfc` depending on `require_nfc`, validates `discard_ranges` invariants (sorted, non-overlapping, in-bounds in codepoints). For the transitional period (Open Question Q3), include a `range_units: Literal["codepoint", "byte"]` parameter that converts byte offsets → codepoint offsets via UTF-8 indexing when set to `"byte"`.
  *Verify:* Unit test loads 4 hand-crafted fixtures (clean article, one-range article, multi-range article, NFD article) and asserts the parsed structures match expected; a fixture with overlapping ranges raises a clear error; a fixture with NFD text under `require_nfc=True` raises, and under `require_nfc=False` warns and normalizes.

- [ ] **TASK-008** [REQ-002] Add a round-trip test: for each fixture article, applying its `discard_ranges` (i.e. removing those codepoint spans from `markdown_text`) yields a "kept" string identical to a checked-in reference.
  *Verify:* `uv run pytest tests/cleaner/test_loader.py -k round_trip` passes on ≥ 5 fixtures.

- [ ] **TASK-009** [REQ-004, REQ-013] Implement `data/align.py::tokenize_and_align(article, tokenizer) -> TokenizedExample`. Use the tokenizer's char-offset mapping (HF fast tokenizers expose `return_offsets_mapping=True`) and label tokens per the rule in design.md § 2.5: a token labeled `drop` iff its codepoint interval is entirely inside some `discard_range`; boundary-straddling tokens labeled `keep` (NFR-002 precision bias).
  *Verify:* Property test on each fixture confirms (a) drop-labeled tokens' codepoint intervals are fully inside some `discard_range`; (b) reconstructing codepoint ranges from contiguous drop-token runs reproduces original `discard_ranges` within ±1 codepoint per boundary.

- [ ] **TASK-010** [REQ-008, NFR-001, NFR-007] Implement `data/split.py::make_splits(item_ids, seed, train_frac=0.70, val_frac=0.15) -> SplitsManifest`. Uses a hash on `(seed, item_id)` to deterministically bucket items, so re-pulled corpora route new items into partitions without disturbing existing ones.
  *Verify:* Unit test confirms (a) two runs with the same `seed` and `item_ids` produce byte-identical manifests; (b) adding new `item_ids` and re-running leaves every previously-assigned item in its original partition; (c) the realized fractions are within ±5% of the targets on a 200-item synthetic corpus.

- [ ] **TASK-011** [NFR-005] Implement `just label-stats` as a Hydra entrypoint that loads the corpus and prints: total articles, fraction with empty `discard_ranges`, distribution of range counts per article, distribution of range lengths in codepoints.
  *Verify:* Running against the fixtures prints sane numbers (e.g. "4 articles, 1 clean, 3 with ranges, mean ranges/article = 1.5").

---

## Phase 3 — Model interface & training loop

Goal: the architecture-agnostic infrastructure that any model implementation will plug into. The first concrete model lands in Phase 4.

- [ ] **TASK-012** [REQ-006, NFR-008] Define `models/base.py::TokenTagger` Protocol with the four methods from design.md § 4.1 (`fit`, `predict_token_probs`, `save`, `load`) plus a `TrainingCallbacks` dataclass with `on_batch_end(step, loss)` and `on_epoch_end(epoch, val_metrics)` hooks.
  *Verify:* `mypy` (or `pyright`) confirms the Protocol; a do-nothing dummy implementation passes `isinstance(..., TokenTagger)` via `@runtime_checkable`.

- [ ] **TASK-013** [REQ-005, REQ-006, NFR-008, NFR-009] Implement `training/loop.py::train(cfg: DictConfig) -> RunResult`. Reads the Hydra config; dynamically imports `cfg.model.module` and instantiates its `TokenTagger`; loads data via Phase 2 modules; calls `tagger.fit(train, val, cfg.model, callbacks)`. No mlflow logging yet (TASK-018 wires it). All paths derive from `HydraConfig.get().runtime.output_dir` — no shared mutable paths (NFR-008).
  *Verify:* Integration test using a stub `TokenTagger` (that just sets `fit_called=True` and emits dummy probs) confirms the loop wires data → fit → predict → returns a `RunResult` with the expected fields.

---

## Phase 4 — First concrete model (HuggingFace encoder)

Goal: ship one architecture so we can do an end-to-end training run. Other architectures (CRF, LoRA seq-tagger) follow the same pattern in later phases / future versions and are out of scope for v1.

- [ ] **TASK-014** [REQ-005] Add `configs/model/encoder_hf.yaml` per design.md § 6 (`name: encoder_hf`, `module: alchimiste.cleaner.models.encoder_hf`, `hf_model_name: distilbert-base-uncased` as the default — revisit during end-to-end validation).
  *Verify:* `uv run python -c "from hydra import compose, initialize; initialize(config_path='../configs', version_base=None); cfg = compose('config'); assert cfg.model.module == 'alchimiste.cleaner.models.encoder_hf'"` exits 0.

- [ ] **TASK-015** [REQ-005, REQ-006] Implement `models/encoder_hf.py::EncoderTagger` wrapping `AutoModelForTokenClassification` with a 2-class head. Implements `fit` (with the two callbacks), `predict_token_probs` (returns softmax probs over the drop class), `save` (HF `save_pretrained` + a `tokenizer_config.json`), and `load`.
  *Verify:* Unit test trains for 1 epoch on 5 fixture articles, then `save → load → predict_token_probs` yields identical per-token probabilities before and after the round-trip (within float tolerance).

- [ ] **TASK-016** [REQ-007, NFR-002] Add weighted cross-entropy support to `EncoderTagger.fit`, driven by `cfg.training.class_weight_drop`. The weight is plumbed into `torch.nn.CrossEntropyLoss(weight=...)` for the drop vs keep classes.
  *Verify:* Training with `class_weight_drop=5.0` on a small but imbalanced fixture produces a model whose drop-class recall on the train set is strictly higher than the same run with `class_weight_drop=1.0`.

---

## Phase 5 — Evaluation

Goal: IoU-based metrics that gate the artifact, plus the failure dump that round-trips into alambic.

- [ ] **TASK-017** [REQ-009] Implement `eval/iou.py::iou_metrics(true_ranges, pred_ranges, thresholds=(0.5, 0.9, 1.0)) -> dict`. For each predicted range, finds the max-IoU ground-truth range; at each τ, counts TP / FP / FN and computes precision / recall / F1.
  *Verify:* Unit test on synthetic predictions where the ground truth is known confirms each metric matches a hand computation. Include the exact-match (τ = 1.0) and lenient (τ = 0.5) edge cases.

- [ ] **TASK-018** [REQ-015, NFR-007] Implement `training/mlflow_io.py` with helpers: `start_run(cfg, oxen_meta)` (creates run, sets tags from design.md § 5.4), `log_batch(step, loss)`, `log_epoch(epoch, val_metrics)`, `log_final(test_metrics, artifact_dir)`. Wire these into `training/loop.py` and a default `TrainingCallbacks` implementation.
  *Verify:* Integration test using mlflow's `file://` tracking URI to a tmp_path: runs the stub-tagger loop, opens the resulting run via `mlflow.tracking.MlflowClient`, asserts that `train/loss` has a time series with ≥ 1 step, that `alchimiste.dataset.oxen_commit` is set as a tag, and that all expected end-of-run metrics are present.

- [ ] **TASK-019** [REQ-010] Implement `eval/failures.py::write_failures(test_results, dst)` producing `failures.jsonl` with the schema from design.md § 7.4 (`item_id`, `content_sha256`, `true_drop_ranges`, `pred_drop_ranges`, `false_positive_ranges`, `false_negative_ranges`, `kept_diff`).
  *Verify:* On a fixture where the model is forced to predict the wrong ranges, the failures file contains exactly that article with the expected FP/FN decomposition and a non-empty `kept_diff`.

- [ ] **TASK-020** [NFR-002, REQ-007] Implement threshold selection in `training/loop.py`: after training, run `predict_token_probs` on the validation set, decode at thresholds τ ∈ {0.30, 0.35, …, 0.90}, compute IoU precision/recall at IoU=0.5 for each τ, pick the lowest τ that meets `cfg.eval.precision_floor` (default 0.85). Fall back to the max-precision τ if none meets the floor and tag `alchimiste.threshold.fell_back_to_max_precision = true` on the mlflow run. Persist the chosen τ to `threshold.json`.
  *Verify:* Unit test with synthetic per-token probabilities confirms the selected τ meets or maximally approaches the floor; the fallback tag fires when no τ meets it.

- [ ] **TASK-021** [REQ-003] Add the clean-only and excessive-ranges metrics to the test-set evaluator and log them to mlflow as `test/clean_articles_kept_clean_fraction` and `test/excessive_ranges_fraction`. Threshold for "excessive" comes from `cfg.eval.excessive_ranges_threshold` (default 6).
  *Verify:* On fixtures with known per-article range counts, the reported fractions match a hand count.

- [ ] **TASK-022** [REQ-009, NFR-006] Wire `just eval` as a Hydra entrypoint that loads an mlflow run id, fetches its artifact, runs IoU + failure dump against the test split deterministically.
  *Verify:* `just eval run=<id>` produces identical `metrics.json` across two consecutive invocations (`diff` is empty).

---

## Phase 6 — Inference + mlflow artifact

Goal: ship the model and the code that runs it as a single mlflow model that any consumer can load.

- [ ] **TASK-023** [REQ-012, REQ-013] Implement `inference/decode.py::decode_token_runs(probs, codepoint_offset_mapping, threshold, min_run=0) -> list[tuple[int, int]]` per design.md § 4.3 — threshold → group → emit `[min(cp_start), max(cp_end))` → filter → sort and merge.
  *Verify:* Property test: outputs are sorted, non-overlapping, within `[0, len(text)]` for 10 randomly-generated probability arrays.

- [ ] **TASK-024** [REQ-011, REQ-012, REQ-013] Implement `inference/pyfunc.py::CleanerModel(mlflow.pyfunc.PythonModel)` with `load_context` (loads the encoder weights, tokenizer, threshold) and `predict` (NFC-normalizes input → tokenize → forward → decode → returns `{"drop_ranges": [...]}`). Predicts accept either a single string or a list.
  *Verify:* Unit test instantiates `CleanerModel`, manually loads context from a tmp_path artifact, runs `predict` on a fixture string in both NFC and NFD form, and confirms identical output ranges.

- [ ] **TASK-025** [REQ-011, NFR-004] Implement `training/mlflow_io.py::log_pyfunc_model(artifact_dir, run)` that assembles the artifact layout from design.md § 8 (`model/`, `config.yaml`, `threshold.json`, `splits.json`, `metrics.json`, `failures.jsonl`, `inference/` with `pyfunc.py`, `decode.py`, `align.py`, `normalize.py`) and calls `mlflow.pyfunc.log_model(..., python_model=CleanerModel(), code_path=[...], registered_model_name=cfg.mlflow.model_name)`. Emits a warning if the assembled artifact exceeds 2 GB.
  *Verify:* Integration test runs a 1-epoch training on fixtures, registers the model under a test name in a tmp `file://` mlflow registry, then `mlflow.pyfunc.load_model(...)` from a fresh process (using `subprocess.run` to isolate) produces predictions on a fixture article — confirming the artifact is genuinely self-contained per REQ-011.

- [ ] **TASK-026** [REQ-012, NFR-005] Implement `inference/cli.py` that reads stdin, calls the same code path as `CleanerModel.predict` (importing it directly so the two stay in lockstep), writes `{"drop_ranges": [...]}` to stdout. Wire `just predict run=<id>` to fetch the artifact from mlflow and pipe stdin through.
  *Verify:* `echo "$(cat fixtures/sample_article.md)" | just predict run=<id>` emits valid JSON; `jq .drop_ranges` returns an array; output matches what `CleanerModel.predict` returns when invoked directly on the same text.

- [ ] **TASK-027** [NFR-003] Add an optional `--bench` flag to `inference/cli.py` that times a single forward pass over a representative article and prints wall time.
  *Verify:* On a developer laptop CPU, `--bench` on a 5-KB fixture reports < 60 s with the default encoder; fail the test (with a clear message) if > 300 s (NFR-003 hard cap).

- [ ] **TASK-028** [NFR-008] Wire `just train` and `just train-multi` to the Hydra entrypoint (`uv run python -m alchimiste.cleaner.training.loop`). `train-multi` forwards `-m` to Hydra for multirun. Confirm that two simultaneous `just train` invocations (different `model=` overrides or `seed=` overrides) complete without colliding.
  *Verify:* In a test harness, spawn two `just train` subprocesses in parallel on fixtures with different `seed` overrides; both exit 0; both produce distinct mlflow run ids; neither writes to a path that the other also writes to (assertion: their resolved Hydra `outputs/<timestamp>/` paths are disjoint).

---

## Phase 7 — End-to-end validation

Goal: train on the real labeled corpus, confirm the v1 gates, and commit the run notes.

- [ ] **TASK-029** [REQ-001, REQ-005, REQ-007, REQ-009, REQ-011, REQ-014, REQ-015] Pull the latest `cleaning` dataset via `alchimiste pull cleaning <oxen-dir>`, then run `just train` against it with the default config. Commit the resulting run's `metrics.json`, `threshold.json`, and a short notes file under `docs/cleanup-model/runs/<date>.md` including the mlflow run URL and the source oxen commit hash.
  *Verify:* The mlflow run shows `test/iou_precision@0.5 ≥ 0.85` **and** `test/iou_recall@0.5 ≥ 0.70` (REQ-007); `test/clean_articles_kept_clean_fraction ≥ 0.95` and `test/excessive_ranges_fraction ≤ 0.05` (REQ-003); the run is tagged with the source oxen commit hash.

- [ ] **TASK-030** [NFR-001] Re-run TASK-029's training twice with the same config and seed; confirm test-set IoU-F1@0.5 differs by ≤ 0.5 points across runs on the same machine.
  *Verify:* Diff of the two runs' `metrics.json` is within tolerance; both runs share the same source oxen commit hash tag.

- [ ] **TASK-031** [NFR-008] Run two `just train` invocations in parallel with different `model.architecture_params.hf_model_name` overrides (e.g., DistilBERT vs MiniLM if both are configured). Both complete successfully; both appear as distinct runs under the same mlflow experiment.
  *Verify:* `mlflow.search_runs(...)` returns both runs with different `alchimiste.model.architecture` (or hf model name) tags and disjoint output directories.

- [ ] **TASK-032** [REQ-010] Review `failures.jsonl` from the TASK-029 run; tag each failure as (a) labeling error, (b) model error, or (c) genuinely ambiguous. Feed (a) and (c) back to the labeler via alambic (manual for v1; the formal ingest hook is Open Question Q4). Add a short triage table to the run's notes file.
  *Verify:* The notes file under `docs/cleanup-model/runs/<date>.md` contains the triage breakdown with counts per category.

---

## Notes on sequencing

- Phases 2 and 3 can overlap once TASK-009 lands — the `TokenTagger` Protocol does not need real data to compile against.
- Phase 4 (concrete encoder) depends on the Protocol from TASK-012; both Phase 5 metric work and Phase 6 inference work can begin in parallel with Phase 4 if the stub tagger from TASK-013 is sufficient for early integration.
- Phase 6 mlflow artifact assembly (TASK-025) depends on Phase 5's threshold selection (TASK-020) — the threshold ships inside the artifact.
- Phase 7 is the only phase that requires the real corpus pulled via `alchimiste pull`; everything before runs on the small synthetic fixtures.

## Cross-repo coordination items (not v1 tasks)

These belong on the alambic side and are tracked here for visibility but are not blocking v1 of the cleaner:

- **Codepoint offsets + NFC adoption in alambic** (REQ-013 / Open Question Q3). Until this lands, TASK-007 runs in transitional `range_units="byte"` mode.
- **failures.jsonl ingest endpoint in alambic** (REQ-010 / Open Question Q4). Until this lands, TASK-032 is a manual review.

## Out-of-scope reminders (from requirements.md § Out of Scope)

- No streaming / long-doc chunking, no multilingual handling, no paraphrasing, no auth, no mlflow tracking-server provisioning. Resist scope creep into these areas during implementation.
