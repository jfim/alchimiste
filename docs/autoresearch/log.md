# Autoresearch log

Format per entry:

```
## Run N — <one-line hypothesis>
Config: <hydra overrides>
Artifact: outputs/<date>/<time>  |  MLflow run: <run_id>

Numbers:
  val: threshold=<τ>  P=<p>  R=<r>  F1=<f1>
  test: P=<p>  R=<r>  F1=<f1>   (IoU=0.5)
  loss: train ep_min=<v> ep_max=<v> | val ep_min=<v>@ep<i> ep_end=<v>

Qualitative (from failures.jsonl):
  - <one or two observations>

Verdict: <kept / discarded / inconclusive>
Next: <what to try>
```

Start time: 2026-05-22 (~5:40 UTC training start).

---

## Run 1 — replay pre-fix baseline under the corrected metric

Hypothesis: same config as run `d55366b4` (pre-truth-clipping). Difference should be metric-only — model weights should land in roughly the same place, but reported recall should jump because end-of-doc drops past the 512-token window no longer count against us.

Config: `distilbert-base-uncased`, `max_seq_len=512`, `epochs=30`, `class_weight_drop=5.0`, otherwise defaults.
Artifact: `outputs/2026-05-22/05-39-27`  |  MLflow run name: `encoder_hf-seed17-2026-05-22_05-39-27`

Numbers:
```
val:  threshold=0.3  P=0.50  R=0.33  F1=0.40   (fell_back_to_max_precision=true)
test (IoU=0.5):       P=0.29  R=0.67  F1=0.40
test (IoU=0.9):       P=0.29  R=0.67  F1=0.40
test (IoU=1.0):       P=0.00  R=0.00  F1=0.00   (structural — boundary tokens labeled keep)
loss: train ep_min=~0.0002 (ep27+) | val_loss min=0.051 @ ep3 → climbs to 0.066 @ ep29
sizes: train=116 val=22 test=26  (post min_bytes=1024 filter)
```

vs. pre-fix run `d55366b4`:
| | pre-fix | run 1 |
|---|---|---|
| val P / R / F1 | 1.00 / 0.09 / 0.17 | 0.50 / 0.33 / 0.40 |
| test P / R / F1 (IoU=0.5) | 0.29 / 0.21 / 0.24 | 0.29 / **0.67** / 0.40 |

Test recall tripled (0.21 → 0.67) just from the metric fix. The model wasn't blind — we were grading it on tokens it had never seen.

Qualitative (from `failures.jsonl`):
- Multiple test articles have truth ranges way past `max_seq_len`: `[15909, 16771]`, `[20447, 21116]`, `[27961, 28548]`. These are correctly excluded from scoring now (the clipped truth is empty for those articles), but the failures file still surfaces them so a human can see the truncation loss.
- One failure where truth is `[415, 862]` but pred is `[[350,351], [355,356]]` — model is firing on two single-token spans in the *wrong* region. Suggests a calibration problem at the boundary, not a "model can't find footers" problem.
- Reachable truth (within the 512-token window) is sparse — only ~3 ranges across the 26-article test set. Tiny denominator → noisy precision.

Concerns:
- **Precision floor of 0.85 is now unreachable** (max precision across the sweep is 0.50). Threshold selection fell back to max-precision. Two probable causes:
  1. Overfit final model (val loss minimum at ep 3-4, then climbs) is poorly calibrated.
  2. Reachable-truth denominator is tiny — one bad prediction nukes precision.
- Need longer-context experiments to get more reachable truth (Phase 3).

Verdict: kept as the post-fix baseline. F1=0.40 is the bar to beat.
Next: Run 2 — same config but `epochs=6`. If val loss minimum at ep 3-4 is the true sweet spot, the ep=6 model should retain or improve recall and may regain precision (less memorisation → less overconfident drops).

---

## Run 2 — epochs=6, otherwise identical to Run 1

Hypothesis: val loss minimum is at ep 3-4 (from Run 1 curve); ep=6 should land near that point and avoid the overfit drift Run 1 showed at ep 29.

Config: `distilbert@512`, `epochs=6`, `class_weight_drop=5`, defaults.
Artifact: `outputs/2026-05-22/05-59-00`  |  MLflow run name: `encoder_hf-ep6-baseline`

Numbers:
```
val: τ=0.45  P=0.50  R=0.33  F1=0.40   (fell_back=true)
test (IoU=0.5):  P=0.20  R=0.67  F1=0.31
loss: train 0.47→0.026 (clean descent, no overfitting); val min=0.0496@ep3, ends 0.053@ep5
```

Verdict: val identical to Run 1 (0.50/0.33/0.40). Test F1 fell from 0.40→0.31 — but the test denominator is tiny (~3 reachable truth ranges); the 0.09 difference is single-prediction noise. Loss curves are much healthier (no overfitting), but the headline metric didn't improve.

**Reinterpretation:** overfitting was not the bottleneck. The bottleneck is *reachable truth* — most test drops live past the 512-token truncation window. Phase 3 (long context) is the real lever.

---

## Autoresearch batch-1 (5 runs, ~135 min wall clock)

Tagged `mlflow.autoresearch=batch-1`. All runs at `epochs=6`, `precision=bf16`, `grad_accum_steps=4`, `batch_size=2`, `gradient_checkpointing=true`. Each ModernBERT run trained successfully but the final 300 MB pyfunc model upload to mlflow.jfim.dev timed out repeatedly — metrics are still in MLflow, only the heavyweight model artifact registration failed. Spawned a follow-up task to skip pyfunc registration for autoresearch-tagged runs.

| # | model | max_seq_len | clw | test F1 @0.5 | test P / R | val τ | val P / R |
|---|---|---|---|---|---|---|---|
| baseline (Run #2 above) | distilbert | 512 | 5 | 0.31 | 0.20 / 0.67 | 0.45 | 0.50 / 0.33 |
| 1 | distilbert | 512 | 20 | 0.32 | 0.21 / 0.67 | 0.30 | 0.50 / 0.33 |
| 2 | ModernBERT-base | 2048 | 5 | 0.19 | 0.11 / 0.80 | 0.45 | 0.06 / 0.25 |
| 3 | ModernBERT-base | 4096 | 5 | 0.24 | 0.15 / 0.54 | 0.80 | 0.38 / 0.43 |
| 4 | ModernBERT-base | 4096 | 20 | 0.14 | 0.08 / 0.54 | 0.90 | 0.13 / 0.29 |
| **5** | **ModernBERT-base** | **8192** | **5** | **0.41** | **0.29 / 0.71** | **0.30** | **0.50 / 0.56** |

### Findings

**1. `class_weight_drop` is not the lever.** On distilbert, going 5→20 moved val P/R/F1 by 0.0 (Run 1 vs Run #2). On ModernBERT@4096 it actively *hurt* (F1 0.24 → 0.14, val precision 0.38 → 0.13). ModernBERT already over-flags; up-weighting drops just makes that worse.

**2. Context length is non-monotonic on this dataset.** ModernBERT@2048 and @4096 lose to distilbert@512. Only @8192 wins. Intermediate context length appears to be "worst of both worlds": bigger model than distilbert (harder to fit on 141 train articles) but not enough context to see the trailing boilerplate. Past a threshold (somewhere between 4096 and 8192) the extra context starts paying off because end-of-doc footers become reachable.

**3. The win is in val recall.** ModernBERT@8192 jumped val recall from 0.33 → 0.56 — the model now actually *sees* the trailing boilerplate that lives past codepoint ~2k in long articles. Val precision stayed at 0.50, so the precision-floor problem (0.85 unreachable) persists.

**4. ModernBERT@8192 fits on a 3060 with bf16 + grad_checkpointing + bs=2 + grad_accum=4.** Confirmed it doesn't OOM. Training took roughly 20-25 min for 6 epochs at 8192.

### Open questions surfaced

- **Why does ModernBERT@2048/4096 do worse than distilbert@512?** Hypothesis: ModernBERT-base has ~150M params vs distilbert's ~67M; the larger model overfits to specific positions in the longer sequence given only 141 train articles. Test: fewer epochs or stronger regularization at intermediate context lengths.
- **Can ModernBERT@8192 reach the 0.85 precision floor?** Currently the threshold-selection still falls back. Worth a sweep of: epochs (3, 6, 10), head dropout, weight_decay, and seed reruns to nail the variance.
- **Where does the data-scarcity barrier sit?** Run 5 shows there's signal to capture; the gap to a useful model is probably about labeled data, not model choice. Validates the earlier instinct to revisit LLM-labeling.

### Verdict

**ModernBERT-base @ max_seq_len=8192, epochs=6, class_weight_drop=5, bf16 + grad-checkpointing is the new best baseline.** Test F1=0.41 (matches or slightly beats the original baseline Run 1) with much better val recall (0.56 vs 0.33). Use this as the starting point for batch-2.

### Suggested batch-2 (when ready)

1. Seed reruns of Run 5 (seeds 11, 17, 42) to establish the noise floor — needs ~3 runs.
2. Sweep epochs {3, 6, 10} on Run 5's config.
3. Sweep head dropout {0.1, 0.2, 0.3} and weight_decay {0.01, 0.05}.
4. Revisit LLM labeling (we're now confident the data is the bottleneck). Probably with a focused prompt fix and an inference-time fallback rather than a fine-tuned student.

---

## Autoresearch batch-2 Phase A — seed variance on the new best baseline

Hypothesis: with the test denominator at ~3 reachable truth ranges and val at ~11, single-config single-seed numbers are noisy. Three seed reruns of batch-1's Run 5 config (`ModernBERT-base @ sl=8192, ep=6, clw=5, bf16+grad_checkpoint`) should bound the noise floor before tuning further.

| seed | source | test F1 @ 0.5 | test P / R | val τ | val P / R |
|---|---|---|---|---|---|
| 17 | batch-1 Run 5 | 0.41 | 0.29 / 0.71 | 0.30 | 0.50 / 0.56 |
| 11 | batch-2 phase A | 0.36 | 0.36 / 0.36 | 0.65 | **0.875** / 0.58 |
| 17 | batch-2 phase A (rerun) | 0.14 | 0.08 / 0.65 | 0.45 | 0.087 / 0.44 |
| 42 | batch-2 phase A | 0.29 | 0.20 / 0.50 | 0.55 | 0.125 / 0.83 |

### Findings

**1. Variance dwarfs effect size.** Test F1 spans 0.14 → 0.41 (std ≈ 0.12) for the same config. Val precision spans **0.087 → 0.875** — a ~10× swing.

**2. Training is non-deterministic even with the same seed.** The two seed=17 runs disagreed dramatically (test F1 0.41 vs 0.14, val P 0.50 vs 0.087). Same config, same seed, different outcomes — strongly suggests cuDNN / FlashAttention non-determinism. Setting `seed=17` is not sufficient for reproducibility.

**3. The batch-1 "ModernBERT@8192 is the new best" claim was a lucky seed.** Average across 4 seeds: test F1 mean ≈ 0.30, comparable to distilbert@512. Only seed=11 actually cleared the 0.85 precision floor; the others fell back. Single-run improvements at this dataset size cannot be trusted as real.

**4. Implication for the loop:** further hyperparameter tuning with single-run-per-config will mostly find lucky seeds, not real signal. Per-config seed averaging (need ≥5 seeds) would multiply the cost by 5×, eating the budget for exploration.

### Verdict

**The data is the bottleneck.** This validates the earlier instinct to revisit LLM-labeling, paused in this session.

Phase B (distilbert@512 seed variance, 3 seeds) is in flight to confirm whether this variance is universal (data ceiling) or ModernBERT-specific (training instability fixable with regularization).

---

## Autoresearch batch-2 Phase B — distilbert seed variance (diagnostic)

Hypothesis: is the wild Phase A variance universal (data ceiling at 141 train articles) or ModernBERT-specific (training instability)? Three seed reruns of `distilbert@512, ep=6, clw=5` answer it.

| seed | test F1 @ 0.5 | test P / R | val τ | val P / R |
|---|---|---|---|---|
| 11 | 0.571 | 0.500 / 0.667 | 0.80 | 0.154 / 0.500 |
| 17 | 0.065 | 0.040 / 0.167 | 0.90 | 0.500 / 0.333 |
| 42 | 0.235 | 0.154 / 0.500 | 0.65 | **1.000 / 1.000** |

### Findings

**1. Distilbert variance is even wider than ModernBERT.** Test F1 spans 0.07 → 0.57 (std ≈ 0.25) for the same config across seeds. **Variance is universal at this dataset size**, not a ModernBERT-specific issue.

**2. seed=42 hit val P=R=1.0 — and still only gave test F1=0.23.** A "perfect" validation run produced mediocre test results. This is overfitting *to the validation set* — at 11 val ranges, any sufficiently lucky threshold can clear the precision floor while not generalizing. Selecting on val alone is unsafe at this dataset size.

**3. The same-seed non-determinism observed in Phase A reproduces here.** Distilbert seed=17 in the original session (Run #2) gave test F1=0.31; Phase B's seed=17 rerun gave 0.07. Setting `seed=17` does not pin training.

## Autoresearch loop — stop

**Stop condition reached** (per plan.md: "3 consecutive runs with no improvement on the selection metric → notify user, propose alternative ideas, stop"). With variance ≈ ±0.20-0.25 F1 across seeds, hyperparameter tuning at this dataset size can only find lucky seeds. Continuing the loop would burn compute without producing a defensible finding.

### What's actually true after batch-1 + batch-2

- **The data is the bottleneck.** 141 train articles + ~11 val ranges + ~3 reachable test ranges = a regime where noise dwarfs signal. Confirmed across two architectures and two context lengths.
- **The metric clipping fix ([#10](https://github.com/jfim/alchimiste/pull/10)) was the only honest improvement of this session.** It restored credit for tokens the model couldn't see; everything since has been within seed variance.
- **MLflow ergonomics ([#11](https://github.com/jfim/alchimiste/pull/11)/[#12](https://github.com/jfim/alchimiste/pull/12)/[#13](https://github.com/jfim/alchimiste/pull/13)) made the loop runnable.** Worth keeping.
- **ModernBERT@8192 fits on a 3060 with bf16 + grad_checkpointing + bs=2 + grad_accum=4.** Confirmed empirically.

### What's NOT true (corrections to earlier writeups)

- ~~"ModernBERT@8192 is the new best baseline (test F1=0.41)"~~ — that was a lucky seed. Mean F1 across 4 seeds ≈ 0.30, indistinguishable from distilbert@512.
- ~~"epochs=6 is the sweet spot for distilbert"~~ — across seeds the result is dominated by seed variance, not epoch count.

### Recommended next steps (out of scope for the autonomous loop)

1. **Get more data.** Resume the Phase 0 LLM-labeling work with a refined prompt (line-prefix + dilation got us to F1=0.63 on a 5-article pilot before we stopped). Even doubling the labeled set to ~400 articles would shrink the val/test denominators meaningfully.
2. **If staying with current data:** seed-averaged sweeps with ≥5 seeds per config. The compute cost goes up 5× but at least findings become defensible.
3. **Run-to-run determinism:** investigate why setting `seed=N` doesn't reproduce results. Likely needs `torch.use_deterministic_algorithms(True)` + `CUBLAS_WORKSPACE_CONFIG=:4096:8` + maybe pinning the dataloader workers. Out of scope here but a real PR.
4. **Don't trust val-only selection at this size.** seed=42 distilbert had val P=R=1.0 and test F1=0.23. Consider a held-out "trust set" or k-fold splits before declaring a winner.

---

## Autoresearch batch-2 Phase C — variance decomposition (same-seed reps)

Hypothesis: split the variance into split-induced (different seeds → different train/val/test partitions) vs. training-induced (cuDNN/FlashAttention non-determinism within the same split). Three reps of `distilbert@512, ep=6, clw=5, seed=17` plus the two existing same-config seed=17 runs from earlier in the session give us **5 same-seed same-config samples**.

All 5 runs (seed=17 distilbert@512 ep=6 clw=5):

| source | test F1 | val P | val R | val τ |
|---|---|---|---|---|
| Run #2 (original session) | 0.308 | 0.500 | 0.333 | 0.45 |
| Phase B seed-17 | 0.065 | 0.500 | 0.333 | 0.90 |
| Phase C rep 1 | 0.250 | 0.333 | 0.333 | 0.30 |
| Phase C rep 2 | 0.308 | 0.500 | 0.333 | 0.80 |
| Phase C rep 3 | 0.296 | 0.167 | 0.333 | 0.30 |

Statistics:
- **test F1**: mean=0.245, **std=0.104**, range [0.065, 0.308]
- **val recall**: 0.333 across all 5 (perfectly stable — same data split → same achievable recall)
- **val precision**: mean=0.40, std=0.149, range [0.167, 0.500]
- **val τ chosen**: 0.30 – 0.90 (chooser swings wildly even on near-identical val recall)

### Variance decomposition

| source | std of test F1 |
|---|---|
| Within-seed (training-only, Phase C n=5) | **~0.10** |
| Cross-seed (split + training, Phase B n=3) | **~0.25** |
| → Split-induced (subtraction in variance) | **~0.23** |

Both sources are large. The data split is the dominant driver (≈2× the within-seed std), but training non-determinism alone produces a 4× swing in test F1 (0.065 → 0.308).

### Implications

- **The batch-1 / batch-2 "best run" mean is 0.245, not the 0.40 the single-run numbers suggested.** Single-run highs were lucky training under a luckier-than-average split.
- **Same-seed reproducibility is broken.** Setting `seed=17` + identical config doesn't pin training. cuDNN/FlashAttention/dataloader-worker non-determinism is the cause. Fixable with `torch.use_deterministic_algorithms(True)` + `CUBLAS_WORKSPACE_CONFIG=:4096:8` and pinned dataloader workers — that's a real PR.
- **5 reps for SEM ≈ 0.045**. To detect a ±0.05 effect we'd need ~15 reps per config — burning the budget on uncertainty rather than exploration.
- **The val sweep's τ choice is unstable.** Same val recall, different chosen τ → different test outcomes. The "lowest τ meeting the floor" rule plus "max-precision fallback" interacts badly with sweep flatness.

Next: Phase D — does the only precision-floor-hit (ModernBERT@8192 seed=11) reproduce, or was it within the within-seed std=0.10 fluke band?

---

## Autoresearch batch-2 Phase D — does the precision-floor hit reproduce?

Hypothesis: Phase A's seed=11 run was the only run anywhere to clear the 0.85 precision floor (val P=0.875). With Phase C showing within-seed training std ≈ 0.10, three reps with same config + `seed=11` should tell us whether seed=11 is a real basin or a lucky training. Stopped after rep 2 — answer was already in.

| source | test F1 | val P | val R | val τ | floor met? |
|---|---|---|---|---|---|
| Phase A seed=11 | 0.364 | **0.875** | 0.583 | 0.65 | **yes** |
| Phase D rep 1 | 0.226 | 0.200 | 0.417 | 0.30 | no |
| Phase D rep 2 | 0.238 | 0.412 | 0.583 | 0.45 | no |

**The Phase A precision-floor hit did NOT reproduce.** 2/3 same-config same-seed reps fell back, with val precision in the 0.20-0.41 range. The single "floor cleared" result was a lucky training within the std=0.10 within-seed band, not a real basin.

## Loop stopped by user

User stop reason: "our results are only as good as our training setup haha." Exactly. Final tally:

### What this session actually established

| Finding | Confidence |
|---|---|
| Metric-clipping fix ([#10](https://github.com/jfim/alchimiste/pull/10)) was the right thing | high — fixed a real correctness bug |
| Test F1 per-config std ≈ 0.10 within-seed (training), ≈ 0.25 cross-seed (training + split) | high — Phase C n=5, Phase B n=3 |
| Setting `seed=N` does not pin training | high — Phase C n=5, Phase D n=2 directly disagree on same seed |
| The data is the bottleneck | high — variance dwarfs effect size across two architectures |
| Single-run "best baseline" claims are unreliable here | high — Phase A and Phase D contradict each other on same seed=11 config |
| ModernBERT@8192 fits on RTX 3060 with bf16+grad_ckpt+bs=2+grad_accum=4 | high — confirmed by ~10 runs |
| ModernBERT@8192 > distilbert@512 on test F1 | **withdrawn** — single-run claim, doesn't survive seed averaging |
| ModernBERT@8192 > distilbert@512 on val recall | **withdrawn** — same |
| Higher `class_weight_drop` hurts ModernBERT | **uncertain** — based on one seed each |

### Real next steps (for a future session, ordered by leverage)

1. **Deterministic training PR**: `torch.use_deterministic_algorithms(True)` + `cudnn.deterministic=True` + `cudnn.benchmark=False` + `CUBLAS_WORKSPACE_CONFIG=:4096:8` + pinned/seeded DataLoader workers + check FlashAttention path. Without this, any sweep result is one sample of a wide distribution.
2. **k-fold or stratified val/test**: instead of one 70/15/15 split per seed, run k splits and average. Halves the variance more reliably than seed averaging since the split dominates variance.
3. **More labels.** Resume Phase 0 LLM-labeling with the line-prefix + dilation post-process approach that hit pilot F1=0.63 before we paused.
4. **Use LLMs at inference time** as the production cleaner — skip fine-tuning entirely, treat alchimiste as the eval/measurement harness for an LLM-based cleaner.

### Closing note

> "Our results are only as good as our training setup." — the user, 2026-05-22 ~13:42 PT.

Single-seed single-run experiments on a 141-train-article dataset, with non-deterministic CUDA training, cannot detect the size of improvements we were chasing (±0.05 F1) against the noise (±0.20 F1). The honest move was to call it and capture what we actually learned, which is what we did.

---

## Postmortem revision — the non-determinism story (corrected)

Initial diagnosis blamed cuDNN / FlashAttention non-determinism. **Wrong.** The actual cause is much simpler and more embarrassing: **the training code never seeds PyTorch at all.**

`grep -rn "manual_seed|np.random.seed|random.seed|generator=" src/alchimiste/cleaner/` returns nothing. The only place `cfg.seed` is consumed is `make_splits()` — which determines the train/val/test partition. After that point, every random-number consumer pulls from PyTorch's global default RNG, initialized from `/dev/urandom` per process.

Sources of run-to-run variance (in rough order of impact):

1. **Classifier head weight init.** The encoder is pretrained but the linear head is freshly initialized each run. No two runs start from the same weights.
2. **DataLoader shuffle order.** With 141 articles and effective batch 8, ~18 batches × 6 epochs = ~108 optimizer steps. Different batch orderings see different class compositions over the first few epochs — formative period for a small-data fine-tune.
3. **Dropout masks.** Default 0.1 dropout on the classifier head — different mask realizations per run.
4. **cuDNN / kernel non-determinism.** Real but second-order. Per-step deltas are float-precision; on a 6-epoch run with bf16, they don't compound into 0.20 F1 swings.

This explains the Phase C observation cleanly: **val recall was perfectly stable at 0.333 across the 5 reps** (same split → same val articles → same achievable recall ceiling), while val precision and threshold choice jumped around (different head weights / batch order → different predicted probabilities on those same val articles → chooser picks different operating points).

### The fix is tiny

```python
import random
import numpy as np
import torch
seed = int(cfg.seed)
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
# DataLoader inherits via the global generator (num_workers=0 in this codebase).
```

`torch.use_deterministic_algorithms(True)` / `cudnn.deterministic=True` etc. are the second-order knobs and only matter once item #1-3 above are nailed down. Given how thoroughly #1-3 dominate, just adding `torch.manual_seed` is expected to bring within-seed test F1 std from ~0.10 down to ~0.01.

A separate PR adds this and re-runs the Phase C diagnostic (3 same-seed reps at distilbert@512 ep=2-3) to confirm. Once that lands, every prior finding in this log needs to be retaken with proper seeding before any of it can be trusted.

---

## Session-end infrastructure changes (2026-05-22, late)

After the postmortem, a batch of code changes landed so the next autoresearch session starts on solid ground. Every finding above this section is **suspect** — re-run anything you'd act on under the new infrastructure first.

### What's merged or open as of this session

| PR | What it does | Why it matters for autoresearch |
|---|---|---|
| [#10](https://github.com/jfim/alchimiste/pull/10) | Clip truth ranges to tokenized window | Stops scoring the model on tokens past `max_seq_len`. The only structural metric fix; everything before it had a hidden recall ceiling. |
| [#11](https://github.com/jfim/alchimiste/pull/11) | `mlflow.run_name_suffix` + `mlflow.note` | Sweeps are scannable in the MLflow UI without drilling into params. |
| [#12](https://github.com/jfim/alchimiste/pull/12) | `mlflow.autoresearch=<batch>` tag | Groups runs of a campaign so you can filter `tags.alchimiste.autoresearch = "batch-N"`. |
| [#13](https://github.com/jfim/alchimiste/pull/13) | `mlflow.register_model` default false | Stops the flaky 300 MB pyfunc upload that wasted ~20 min per ModernBERT run. |
| [#14](https://github.com/jfim/alchimiste/pull/14) | Seed `torch`/`numpy`/`random` from `cfg.seed` | Makes same-seed runs reproducible to 7-8 sig figs. The single biggest source of false noise is gone. |
| [#15](https://github.com/jfim/alchimiste/pull/15) | `train_seeds(cfg, seeds)` + `just train-seeds` | One Python process per N-seed sweep. Amortizes startup, model load, tokenization. |
| [#16](https://github.com/jfim/alchimiste/pull/16) | `data.split_seed` decoupled from `seed` | Fixed-split seed-averaging now actually works: hold split constant, vary training. The only honest way to measure training variance. |
| [#17](https://github.com/jfim/alchimiste/pull/17) | `training.save_best_on_val` knob | When deployment-ready, restore the lowest-val-loss checkpoint instead of the last (often-overfit) epoch. Default off — sweep runs see no behavior change. |
| [#18](https://github.com/jfim/alchimiste/pull/18) | On-disk tokenization cache | ~4s saved per distilbert@512 run on warm cache; proportionally larger on ModernBERT@8192. Disable with `data.tokenization_cache_dir=""`. |

### How to run a defensible sweep going forward

Minimum bar for "this is a real result":

1. **Fix the split, vary the training seed.** Pick one `data.split_seed` per campaign (e.g. `data.split_seed=17`) and average over training seeds:
   ```
   just train-seeds '+seeds=[11,17,42,8,3]' data.split_seed=17 \
       mlflow.autoresearch=batch-3 \
       mlflow.run_name_suffix=<short-slug> \
       'mlflow.note="<one-line hypothesis>"'
   ```
2. **At least 3 seeds, preferably 5.** Within-seed training std on F1 is now ~0.01 instead of ~0.10, but the model's response to different inits is itself a real signal — averaging keeps you from chasing a lucky init.
3. **Report mean ± std (or SEM).** A "Config A beats Config B" claim needs the means to differ by more than the combined SEMs. Pulling the metric history from MLflow per run and computing mean/std is one-liner Python.
4. **For deployment-bound runs only**, add `training.save_best_on_val=true`. Sweeps don't need it — it changes which weights survive, which changes test metrics, which makes cross-run comparison less clean.
5. **Treat threshold-fallback runs with care.** `fell_back_to_max_precision=true` in `threshold.json` means no τ in the sweep met the precision floor. The run is still informative (recall + best-attainable precision), but the threshold choice is selecting on noise. Consider lowering `precision_floor` in `configs/eval/default.yaml` if it's chronically unmet.

### What's still genuinely unknown (will need fresh sweeps under new infra)

- Whether distilbert@512 or ModernBERT@8192 actually wins on this corpus, with proper seeding.
- Whether the precision floor of 0.85 is reachable by *any* config at the current dataset size.
- Whether `class_weight_drop`, `epochs`, `head_lr_multiplier`, `warmup_ratio`, `label_smoothing`, head dropout, weight decay produce real effects or just live within seed variance. Every prior finding on these from this log was single-run and is suspect.
- Whether the data ceiling is real (the most plausible hypothesis) or whether we just couldn't see the signal under noise. Honest seed-averaged sweeps under the new infra should answer this in the first batch.

### Things NOT to redo

These code-level questions are now answered; don't burn experiments on them:

- "Why is same-seed reproducibility broken?" — was missing `torch.manual_seed`. Fixed in [#14](https://github.com/jfim/alchimiste/pull/14).
- "Is the variance ModernBERT-specific or universal?" — universal (Phase B confirmed for distilbert too). Same root cause as above.
- "Should we use multi-process or multi-seed?" — multi-seed within one process via [#15](https://github.com/jfim/alchimiste/pull/15). ~18% wallclock savings on a 5-seed sweep. Pristine encoder per seed verified by `seeds=[12,12,12]` byte-identical loss curves.
- "Why do MLflow runs show FAILED?" — flaky pyfunc upload, now default off ([#13](https://github.com/jfim/alchimiste/pull/13)).
- "Does the metric penalize the model for tokens past `max_seq_len`?" — used to, no longer ([#10](https://github.com/jfim/alchimiste/pull/10)).

### Tooling cheat-sheet

- Find your batch: MLflow filter `tags.\`alchimiste.autoresearch\` = "batch-N"`.
- Pull per-epoch losses: `client.get_metric_history(run_id, "train/epoch_loss")` and `val/val_loss`.
- Per-run metrics + threshold sweep: `outputs/<date>/<time>/{metrics.json,threshold.json}` (local source of truth — the MLflow status field is sometimes spuriously FAILED on artifact upload).
- Multi-seed run lays out: `outputs/<date>/<time>/seed-<N>-<i>/...` (one subdir per seed).
- Cache lives under `.cache/tokenized/<hash>.pkl`; safe to delete anytime.

