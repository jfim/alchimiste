# Autoresearch plan

## Objective
Maximize `val/selected_threshold_recall` subject to `precision >= 0.85` (user preference: "better to leave an ad than lose legitimate content"). Tie-break by precision. Use val for selection; test only as a final sanity check.

## Hard constraints
- **Hydra overrides only for autoresearch runs themselves** — no source code edits inside the loop. Supporting infrastructure improvements (metric fixes, MLflow ergonomics, etc.) go in separate PRs *between* batches, with the user's review, and are out of scope for the autonomous loop.
- One run at a time (RTX 3060, 12 GB VRAM).
- Wall-clock budget: run until **2026-05-23 15:00 local**, or until the user says stop.
- Notify on stop or blocker via `claude-notify`.

## Per-batch conventions

Each autoresearch batch gets its own tag value so runs are grouped in the MLflow UI:

```
just train <overrides> \
    mlflow.autoresearch=batch-N \
    mlflow.run_name_suffix=<short-slug> \
    mlflow.note="<one-line hypothesis or knob description>"
```

Examples of useful suffixes: `ep6-clw20-distil-sl512`, `ep6-clw5-modernbert-sl4096`. Keep them under ~30 chars; encode the knobs that vary, not the ones that don't.

Filter the MLflow UI with `tags.\`alchimiste.autoresearch\` = "batch-N"` to see one batch, or `!= ""` to see every autoresearch run across batches.

**Hydra override grammar quirks** (learned the hard way):
- Avoid `,`, `:`, `=`, `→` in values — they trigger list-parsing or are interpreted as new overrides. Stick to letters/digits/dash/period/space inside the note.
- Outer-single, inner-double quoting (`'mlflow.note="my note"'`) is the safest way to ship a multi-word value through `just` + bash + Hydra.

## Known structural facts (don't relearn)
- Corpus: 198 articles (train=141, val=25, test=32) after `min_bytes=1024` filter. ~11 positive ranges in val, ~3 reachable ranges in test (post-truncation-clip). Tiny denominators — single-prediction differences can swing reported precision by 0.1+. Confirm any finding with seed reruns before committing to it.
- Token labels: `drop` only if **fully** inside a discard range; boundary tokens → `keep`. Precision-biased by design. `iou_f1_at_1.0` has a structural ceiling < 1.0; don't chase it.
- Drops cluster at **start AND end** of articles. The truth-clipping fix ([#10](https://github.com/jfim/alchimiste/pull/10)) restricts val+test truth to the tokenized window so a model isn't penalized for tokens it never saw.
- **The data is the bottleneck**, not the model (batch-1 finding). distilbert@512, ModernBERT@2048, ModernBERT@4096 all land near or below test F1=0.32. Only ModernBERT@8192 breaks out (test F1=0.41, val recall 0.33→0.56). At 141 train articles, neither more capacity nor more context past 8k is likely to move the needle much further until we add labels.
- MLflow tracking URI: `https://mlflow.jfim.dev/`, experiment `alchimiste-text-cleaner`. Access from Python:
  ```python
  import mlflow; mlflow.set_tracking_uri("https://mlflow.jfim.dev/")
  client = mlflow.MlflowClient()
  ```
- Per-run artifact lives at `outputs/<date>/<time>/`. Key files: `metrics.json`, `threshold.json`, `failures.jsonl`, `.hydra/overrides.yaml`.
- **Known artifact-upload bug**: ModernBERT runs train successfully but the 300 MB pyfunc model upload to mlflow.jfim.dev times out and marks the run "FAILED" in MLflow. Metrics are still logged. Local `outputs/<date>/<time>/metrics.json` is the source of truth. Tracked separately for a fix that skips pyfunc registration for autoresearch-tagged runs.

## Current best baseline (post batch-1)

`ModernBERT-base @ max_seq_len=8192, epochs=6, class_weight_drop=5, precision=bf16, gradient_checkpointing=true, grad_accum_steps=4, batch_size=2`. Test F1=0.41, val P/R=0.50/0.56 at τ=0.30 (still falls back; 0.85 floor not met). Trained on RTX 3060 in ~25 min.

## Search arc (adapt freely)

**Batch-2 candidates** (pick 5-8 of these, sequence by expected leverage):

1. **Seed reruns of the new best** — seeds 11, 17, 42 to establish the noise floor for ModernBERT@8192. Tiny denominators make this important before tuning further.
2. **Epochs sweep on the new best** — {3, 6, 10}. Val loss may bottom earlier or later at 8192 context vs 512.
3. **Regularization** — head dropout {0.1, 0.2, 0.3}, weight_decay {0.01, 0.05}, label_smoothing {0.0, 0.05}.
4. **Warmup ratio** — {0.0, 0.05, 0.1} for ModernBERT (its scheduler benefits more from warmup than distilbert at this scale).
5. **Class weight at long context** — confirm that clw=5 is actually optimal at 8192 (it hurt at 4096). Try {3, 5, 10}.
6. **head_lr_multiplier** — {1, 5} at long context; the original sweep was at 512 and may behave differently.

**Do NOT pursue** without a new hypothesis:
- More distilbert@512 hyperparameter tuning — capped by reachable truth.
- ModernBERT @ 2048 or 4096 — strictly worse than @8192 *and* worse than distilbert@512 (batch-1).
- Heavier class weight on ModernBERT — actively hurt at 4096 (F1 0.24 → 0.14).

**Out of scope for the autonomous loop** (require user-reviewed PRs):
- New head architectures (CRF, BiLSTM+CRF, etc.).
- Focal loss / asymmetric loss / positional features.
- NeoBERT or other new architectures (need a new `configs/model/*.yaml`).
- Best-on-val checkpoint saving (the loop currently saves last-epoch weights).
- LLM-pseudo-labeling pipeline (Phase 0 work — paused; the data is the bottleneck but distilling is hard at the quality we'd need).

## Per-run protocol
1. Decide config from `log.md` (avoid duplicates within and across batches).
2. Form a one-line hypothesis (e.g. "head dropout 0.2 should improve val recall without sacrificing precision because ModernBERT@8192 looks under-regularized").
3. `just train <overrides> mlflow.autoresearch=batch-N mlflow.run_name_suffix=<slug> 'mlflow.note="<hypothesis>"'` to completion.
4. Parse `outputs/<date>/<time>/metrics.json` + `threshold.json`. Treat the MLflow status (sometimes spuriously FAILED on upload timeout) as advisory; trust the local `metrics.json`.
5. Pull `train/epoch_loss` and `val/val_loss` curves from MLflow if the shape matters for the finding.
6. **Read 2-3 entries from `failures.jsonl`** — what is the model missing? Where in the doc? Note in the log.
7. Append to `log.md`: hypothesis → config → numbers → curve shape → qualitative obs → next idea.
8. Every ~5 runs: write a short batch summary in `log.md` and decide whether to keep going, pivot, or stop.

## Stop conditions
- User says stop, or the agreed wall-clock budget hits.
- 3 consecutive runs in a batch with no improvement on `val/selected_threshold_recall` → notify, propose a different batch or out-of-scope idea, stop the loop.
- Any run OOMs → log + try the next config. If 2 in a row OOM, notify and stop.
- Any run takes substantially longer than expected (>3× the prior median in the batch) → notify before continuing.
