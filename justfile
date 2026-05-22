default:
    @just --list

# Install all dependencies (CPU torch wheel; default).
sync:
    uv sync

# Install with the CUDA torch wheel instead of the CPU one. Use on a
# training box with an NVIDIA GPU and driver compatible with CUDA 13.2.
sync-cuda:
    uv sync --no-group cpu --group cuda

# Run lint checks
lint:
    uv run --no-sync ruff check .

# Auto-fix lint issues
fix:
    uv run --no-sync ruff check --fix .

# Format code
fmt:
    uv run --no-sync ruff format .

# Check formatting without writing
fmt-check:
    uv run --no-sync ruff format --check .

# Run tests
test:
    uv run --no-sync pytest

# Run all checks (lint + format + tests)
check: lint fmt-check test

# Remove build artifacts and caches
clean:
    rm -rf .ruff_cache .pytest_cache dist build *.egg-info
    find . -type d -name __pycache__ -exec rm -rf {} +

pull-extraction repo_dir:
    uv run --no-sync alchimiste pull extraction {{repo_dir}}

pull-cleaning repo_dir:
    uv run --no-sync alchimiste pull cleaning {{repo_dir}}

# --- Text cleaner training pipeline ---
# All operational recipes use `uv run --no-sync` so they don't undo a
# `just sync-cuda` by swapping the CUDA torch wheel back to the CPU one.
# Run `just sync` (or `just sync-cuda` on a training box) once before
# using these.

# Train a cleaner model. Extra args are forwarded to Hydra.
train *ARGS:
    uv run --no-sync python -m alchimiste.cleaner.training.loop {{ARGS}}

# Hydra multirun convenience: `just train-multi model=encoder_hf,crf seed=11,17,42`
train-multi *ARGS:
    uv run --no-sync python -m alchimiste.cleaner.training.loop -m {{ARGS}}

# Train the same config across multiple seeds in ONE Python process.
# Amortizes Python startup, HF Hub revalidation, model + tokenizer load,
# and article tokenization across seeds. Each seed gets its own MLflow
# run + artifact subdirectory.
#
# Example:
#   just train-seeds '+seeds=[11,17,42]' training.epochs=3
#
# HF_HUB_OFFLINE is set on the assumption the model is already cached
# locally — first ever pull of a new HF model should use `just train`
# (online) to populate the cache.
train-seeds *ARGS:
    HF_HUB_OFFLINE=1 uv run --no-sync python -m alchimiste.cleaner.training.loop {{ARGS}}

# Evaluate an existing artifact on the test split. `just eval artifact=runs/<ts>`.
eval *ARGS:
    uv run --no-sync python -m alchimiste.cleaner.eval.run {{ARGS}}

# Predict from stdin -> JSON drop_ranges on stdout.
# Example: `cat article.md | just predict --artifact=runs/2026-05-21/12-00-00`
predict *ARGS:
    uv run --no-sync python -m alchimiste.cleaner.inference.cli {{ARGS}}

# Print corpus statistics for the cleaning dataset.
label-stats *ARGS:
    uv run --no-sync python -m alchimiste.cleaner.data.label_stats {{ARGS}}
