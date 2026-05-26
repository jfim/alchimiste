#!/usr/bin/env bash
# Overnight head-architecture sweep — 2026-05-27.
#
# See docs/autoresearch/overnight-2026-05-27.md for the rationale.
# Designed to fit the midnight–~8 AM cheap-electricity window
# (~7.9 h total wall time).
#
# Phase 1: 5 non-CRF head configs × 3 seeds, 30 epochs each (~5.5 h).
# Phase 2: 2 CRF configs × 1 seed, 6 epochs each (~2.4 h).
#
# All runs tagged `mlflow.autoresearch=overnight-2026-05-27` so they
# can be pulled as a batch:
#   client.search_runs(..., filter_string='tag.alchimiste.autoresearch = "overnight-2026-05-27"')
#
# Run names are `neobert-<config>-s<seed>` for easy UI filtering.

set -u
cd /home/jfim/projects/alchimiste

LOG=/home/jfim/projects/alchimiste/scripts/overnight_head_sweep.log
: >"$LOG"

# Held-constant arguments. The split_seed=17 pin holds the
# train/val/test partition fixed across all runs; only the training
# init (and dataloader shuffle) varies with `seed=`.
COMMON=(
    model=neobert
    data.max_seq_len=4096
    data.split_seed=17
    training.device=cuda
    training.freeze_backbone=true
    training.learning_rate=1e-3
    training.batch_size=1
    training.grad_accum_steps=8
    training.precision=bf16
    +training.eval_batch_size=1
    model.head.type=composable
    mlflow.autoresearch=overnight-2026-05-27
)

run() {
    local name="$1"; shift
    echo "=== $(date -Is) starting: $name ===" | tee -a "$LOG"
    local t0=$(date +%s)
    if just train "${COMMON[@]}" mlflow.run_name_suffix="$name" "$@" >>"$LOG" 2>&1; then
        local dt=$(( $(date +%s) - t0 ))
        echo "=== $(date -Is) FINISHED: $name (${dt}s) ===" | tee -a "$LOG"
    else
        local dt=$(( $(date +%s) - t0 ))
        echo "=== $(date -Is) FAILED:   $name (${dt}s) — continuing ===" | tee -a "$LOG"
    fi
}

# ------------------------------------------------------------------ #
# Phase 1 — non-CRF architecture sweep (15 runs, ~5.5 h)             #
# ------------------------------------------------------------------ #
# Ordered cheapest first so partial completion is still useful.
# `conv_stack=1` means a single conv block, so kernel size is the
# only knob varying receptive field. `conv_stack=2` stacks two convs
# with a GELU between (effective ±2 receptive field at k=3).
#
# Each (config, seed) pair logs to its own MLflow run + its own
# `outputs/<date>/<time>/` artifact dir; the runs are completely
# independent.

PHASE1_EPOCHS=30
PHASE1_HEAD_OFF=(
    model.head.crf=false
    training.epochs=$PHASE1_EPOCHS
)
SEEDS=(11 17 42)

# Config 2: k=3 s=1 full
for s in "${SEEDS[@]}"; do
    run "k3s1full-s${s}" \
        "${PHASE1_HEAD_OFF[@]}" \
        model.head.conv_kernel=3 model.head.conv_stack=1 model.head.conv_mode=full \
        seed=$s
done

# Config 3: k=3 s=1 depthwise_separable
for s in "${SEEDS[@]}"; do
    run "k3s1dwsp-s${s}" \
        "${PHASE1_HEAD_OFF[@]}" \
        model.head.conv_kernel=3 model.head.conv_stack=1 model.head.conv_mode=depthwise_separable \
        seed=$s
done

# Config 4: k=3 s=2 full
for s in "${SEEDS[@]}"; do
    run "k3s2full-s${s}" \
        "${PHASE1_HEAD_OFF[@]}" \
        model.head.conv_kernel=3 model.head.conv_stack=2 model.head.conv_mode=full \
        seed=$s
done

# Config 5: k=7 s=1 full
for s in "${SEEDS[@]}"; do
    run "k7s1full-s${s}" \
        "${PHASE1_HEAD_OFF[@]}" \
        model.head.conv_kernel=7 model.head.conv_stack=1 model.head.conv_mode=full \
        seed=$s
done

# Config 6: k=11 s=1 full
for s in "${SEEDS[@]}"; do
    run "k11s1full-s${s}" \
        "${PHASE1_HEAD_OFF[@]}" \
        model.head.conv_kernel=11 model.head.conv_stack=1 model.head.conv_mode=full \
        seed=$s
done

# ------------------------------------------------------------------ #
# Phase 2 — CRF probe (2 runs, ~2.4 h)                                #
# ------------------------------------------------------------------ #
# Only 6 epochs — this is a *signal* test, not a converged result.
# H5: if val_loss drops faster with CRF in these 6 epochs, it's
# worth a follow-up multi-seed sweep at full 30 epochs. If the
# trajectory matches or lags non-CRF, deprioritize CRF.

PHASE2_EPOCHS=6

# Config 7: linear + CRF (purest test of "does structure help")
run "linear-crf-s17" \
    training.epochs=$PHASE2_EPOCHS \
    model.head.conv_kernel=0 \
    model.head.crf=true \
    seed=17

# Config 8: conv k=3 s=1 full + CRF (does CRF compose with smoothing?)
run "k3s1full-crf-s17" \
    training.epochs=$PHASE2_EPOCHS \
    model.head.conv_kernel=3 model.head.conv_stack=1 model.head.conv_mode=full \
    model.head.crf=true \
    seed=17

echo "=== $(date -Is) DONE ===" | tee -a "$LOG"
