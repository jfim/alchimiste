# Overnight head-architecture sweep — 2026-05-27

Single-batch overnight run designed to be executable by a future Claude
session after a clean restart. Read this end-to-end before launching.

The launcher is at `scripts/overnight_head_sweep.sh`. The plan below is
the *why*; the script is the *what*.

## Purpose

Pick the best frozen-backbone head architecture under controlled
conditions, with enough seed coverage to distinguish signal from
init-noise. This is the disciplined follow-up to the single-seed
exploration in the 2026-05-26 session, and the next step before
considering encoder unfreezing.

## What we know going in

- **Backbone:** Frozen NeoBERT @ 4096 with the fixes from 2026-05-26:
  `freqs_cis` re-init after `from_pretrained` (otherwise NaN losses),
  `no_grad` wrapper around the frozen encoder (otherwise 28× autograd
  activation cost), and `eval_batch_size=1` (otherwise the val pass
  tries to broadcast a (B=22, H=12, T=4096, T=4096) attention mask and
  OOMs). All three are needed; do not change.
- **Best single-seed result so far:** Frozen NeoBERT@512 + `conv1d k=3`
  reached IoU F1=0.086 (recall@0.5=0.5, threshold=0.85). The linear
  head at the same setting was 0.0 across all metrics. The improvement
  was qualitative — `excessive_ranges_fraction` 0.65 → 0.12,
  `clean_articles_kept_clean` 0.05 → 0.85 — confirming local smoothing
  fixes per-token fragmentation.
- **Frozen NeoBERT@4096 + linear** (single seed): val_loss ~0.10
  (about half of @512), but IoU F1 still 0.0. Conclusion: longer
  context helps the *representation* but the *output* is still
  fragmented. The lever is the head.
- **1-epoch smoke at @4096** across 9 head variants all train cleanly;
  val_loss at epoch 0 already ranges 0.09–0.32 across architectures.
- **CRF training is 12× slower** than non-CRF (~13 min/epoch at
  L=4096) because the T=4096 forward/backward loop launches 4096
  CUDA kernels per pass. `torch.jit.script` shaves ~7%. `torch.compile`
  hangs indefinitely trying to unroll the loop. The real fix is a
  parallel-scan implementation (log-semiring matmul tree, O(log T)
  depth instead of O(T)), but that's a multi-day project. For tonight
  we use the JIT'd path and budget CRF accordingly.

## Hypotheses tested overnight

| # | Hypothesis | Test |
|---|---|---|
| H1 | Wider conv kernels improve IoU F1 beyond k=3. | k ∈ {3, 7, 11} at s=1, 3 seeds each. |
| H2 | Two stacked k=3 convs (with GELU between) beat one wider k=5 conv at similar receptive field. | `k=3 s=2` vs `k=7 s=1`, 3 seeds each. |
| H3 | Depthwise-separable matches full conv at much lower parameter count. | `k=3 s=1 dwsp` vs `k=3 s=1 full`, 3 seeds each. |
| H4 | The architecture differences we see are real (above seed noise). | Per-config std across 3 seeds gives an SEM estimate. Differences less than 2× SEM are not real. |
| H5 | CRF improves the loss-decrease *rate* in early epochs, signalling whether structured prediction is worth a full sweep in a follow-up session. | 6-epoch probe on linear+CRF and `k=3 s=1 full`+CRF, compared against the matching non-CRF runs at epoch 6. |

H5 is intentionally a weak test — 6 epochs isn't convergence. But if
CRF drops val/val_loss noticeably faster than non-CRF in those 6
epochs, that's evidence the structured prior is doing useful work and
the full sweep is worth its compute cost. If CRF is the same or worse,
deprioritize CRF for the next session.

## Sweep design

**Held constant:**
- Backbone: `chandar-lab/NeoBERT` (frozen)
- Context: `data.max_seq_len=4096`
- Split: `data.split_seed=17` (so all runs see the same train/val/test partition; only init varies)
- LR: `1e-3` (calibrated for frozen+linear-head in prior session — works across the conv variants too at the 1-epoch smoke)
- Precision: `bf16`
- Batch: `bs=1, grad_accum=8` (effective batch 8 at the memory ceiling)
- Eval batch: `1` (mask OOM avoidance)
- Class weight: `class_weight_drop=5.0` (unchanged from prior runs)
- Epochs: 30 for non-CRF, 6 for CRF (rationale above)

**Phase 1 — head architecture (≈5.5 h, 15 runs):**

| # | Head config | Conv params | Per-run | × seeds |
|---|---|---|---|---|
| 1 | `linear` (baseline anchor; existing single-seed only) | 0 | — | reuse prior |
| 2 | `conv_kernel=3 conv_stack=1 conv_mode=full` | 1.77 M | ~22 min | 3 |
| 3 | `conv_kernel=3 conv_stack=1 conv_mode=depthwise_separable` | ~600 K | ~22 min | 3 |
| 4 | `conv_kernel=3 conv_stack=2 conv_mode=full` | 3.54 M | ~22 min | 3 |
| 5 | `conv_kernel=7 conv_stack=1 conv_mode=full` | 4.13 M | ~22 min | 3 |
| 6 | `conv_kernel=11 conv_stack=1 conv_mode=full` | 6.49 M | ~22 min | 3 |

Seeds: `[11, 17, 42]`. (`17` is our canonical seed; the other two
extend it.) Split-seed pinned to `17` so the data partition is
identical across all runs.

Runs are ordered cheapest-first within Phase 1 so we accumulate
information early if the night runs short.

**Phase 2 — CRF probe (≈2.4 h, 2 runs at 6 epochs):**

| # | Head config | Per-run | × seeds |
|---|---|---|---|
| 7 | `linear + crf` | ~75 min | 1 (seed=17) |
| 8 | `conv_kernel=3 conv_stack=1 conv_mode=full + crf` | ~75 min | 1 (seed=17) |

These run *last* so if Phase 1 overruns we still have the architecture
data.

**Total wall clock:** ~7.9 hours. Fits the midnight–~8 AM window.

## Execution

```bash
bash scripts/overnight_head_sweep.sh
```

All runs are tagged `mlflow.autoresearch=overnight-2026-05-27` so they
can be pulled as a batch from MLflow:

```python
runs = client.search_runs(
    exp.experiment_id,
    filter_string='tag.alchimiste.autoresearch = "overnight-2026-05-27"',
    max_results=50,
)
```

The launcher uses `mlflow.run_name_suffix=<config>-s<seed>` so individual
runs are identifiable in the UI without digging into params.

If the launcher errors on a single run it logs the failure and
continues — one OOM or transient MLflow blip shouldn't kill the
whole batch. The log lives at `scripts/overnight_head_sweep.log`.

## Analysis recipe (for next-session Claude)

1. **Pull all overnight runs** from MLflow using the tag filter above.
   Confirm 17 runs landed (15 Phase 1 + 2 Phase 2). If fewer, check
   the launcher log for failures.

2. **Phase 1 analysis — architecture comparison.** For each of the 5
   non-baseline configs, compute:
   - Mean and std of `test/iou_f1_at_0.5` across the 3 seeds.
   - Mean of `test/excessive_ranges_fraction` and
     `test/clean_articles_kept_clean_fraction`.
   - Mean of `val/val_loss` at the final epoch.

   The single existing linear@4096 result (from the 2026-05-26 session)
   stays as the anchor — no need to re-run.

   **Resolution rule:** a difference between two configs is *real* if
   it exceeds 2 × max(SEM_A, SEM_B). Anything smaller is within seed
   noise.

3. **Hypothesis verdicts** — for each of H1–H4, write a one-line
   verdict (kept / withdrawn / inconclusive) with the numerical
   justification, in the style of `docs/autoresearch/log.md`. Add the
   entry as a new section at the bottom of `log.md`.

4. **Phase 2 analysis — CRF early-trajectory.** Compare val/val_loss
   curves at epochs 1–6 of the two CRF runs against the matching
   non-CRF runs (linear and `k=3 s=1 full` at seed=17) from Phase 1.
   Use `client.get_metric_history(run_id, "val/val_loss")` to pull
   the per-epoch series. Verdict on H5:
   - If CRF reaches the same val_loss in materially fewer epochs:
     CRF is promising → propose a full multi-seed CRF sweep in a
     follow-up session.
   - If CRF lags or matches: CRF is not the lever right now →
     deprioritize, focus next investment on encoder unfreezing or
     more data.

5. **Write a fresh roadmap** at `docs/autoresearch/roadmap-YYYY-MM-DD.md`
   reflecting the new state of the world. Supersede this file rather
   than editing in place. Record the head-architecture winner, the
   noise floor on test F1, and the recommended next step.

## Failure modes and recovery

- **OOM on a Phase 1 run.** Most likely cause is a long-tail article
  hitting an unforeseen memory peak. The launcher continues past the
  failure; that one (config, seed) data point is just missing.
  Mention in the writeup; don't re-run unless the pattern is
  systematic.
- **MLflow tracking-server down.** The launcher will fail every run
  during start_run setup. If this happens, kill the launcher, restart
  MLflow, and resume by editing the launcher to skip already-completed
  configs (each one is logged separately). The artifact dirs under
  `outputs/2026-05-27/*/` are independent of MLflow — runs still
  produce local artifacts even if logging fails (depending on where
  in the pipeline the connection drops; investigate per-case).
- **The flash_attn install drifted / broke.** Re-run
  `uv sync --no-group cpu --group cuda` once before launching.
- **NeoBERT custom-code hash changed on the Hub.** Cached modules at
  `~/.cache/huggingface/modules/transformers_modules/chandar_hyphen_lab/NeoBERT/<sha>/`
  are pinned per snapshot; if a new snapshot is pulled, the
  `freqs_cis` re-init code in `encoder_hf.py:_repair_rope_buffer` may
  no longer find the right `.rotary` submodule path. Re-verify the
  fix still works (one quick forward at L=4096 should produce
  non-NaN output).

## What this run does NOT test

For context, in case the analysis tempts us to over-generalize:

- **Backbone unfreeze.** Separate session. Way more compute.
- **More data / pseudo-labels.** Outside the head-architecture question.
- **Loss tuning** (class_weight_drop, focal, label smoothing). Held
  constant deliberately to isolate the head variable.
- **Long-tail chunking.** 10% of articles exceed 8192 tokens even
  post-filter; truncation past L=4096 is a real but separate concern.
- **Real (multi-seed) CRF sweep.** H5 is a cheap probe. If it
  promises, the follow-up gets the full treatment.
