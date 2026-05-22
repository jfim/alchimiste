default:
    @just --list

# Install all dependencies including dev group
sync:
    uv sync

# Run lint checks
lint:
    uv run ruff check .

# Auto-fix lint issues
fix:
    uv run ruff check --fix .

# Format code
fmt:
    uv run ruff format .

# Check formatting without writing
fmt-check:
    uv run ruff format --check .

# Run tests
test:
    uv run pytest

# Run all checks (lint + format + tests)
check: lint fmt-check test

# Remove build artifacts and caches
clean:
    rm -rf .ruff_cache .pytest_cache dist build *.egg-info
    find . -type d -name __pycache__ -exec rm -rf {} +

pull-extraction repo_dir:
    uv run alchimiste pull extraction {{repo_dir}}

pull-cleaning repo_dir:
    uv run alchimiste pull cleaning {{repo_dir}}

# --- Text cleaner training pipeline ---
# These recipes are placeholders until the corresponding tasks land
# (TASK-013/022/026/028/011 wire them to real Hydra entrypoints).

# Train a cleaner model. Extra args are forwarded to Hydra.
train *ARGS:
    @echo "TODO TASK-028: wire just train -> uv run python -m alchimiste.cleaner.training.loop {{ARGS}}"

# Hydra multirun convenience: `just train-multi model=encoder_hf,crf seed=11,17,42`
train-multi *ARGS:
    @echo "TODO TASK-028: wire just train-multi -> uv run python -m alchimiste.cleaner.training.loop -m {{ARGS}}"

# Evaluate an existing artifact on the test split. `just eval artifact=runs/<ts>`.
eval *ARGS:
    uv run python -m alchimiste.cleaner.eval.run {{ARGS}}

# Predict from stdin -> JSON drop_ranges on stdout.
# Example: `cat article.md | just predict --artifact=runs/2026-05-21/12-00-00`
predict *ARGS:
    uv run python -m alchimiste.cleaner.inference.cli {{ARGS}}

# Print corpus statistics for the cleaning dataset.
label-stats *ARGS:
    uv run python -m alchimiste.cleaner.data.label_stats {{ARGS}}
