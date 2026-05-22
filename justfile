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

# Evaluate an existing mlflow run on the test split.
eval *ARGS:
    @echo "TODO TASK-022: wire just eval -> uv run python -m alchimiste.cleaner.eval {{ARGS}}"

# Predict from stdin -> JSON drop_ranges on stdout.
predict *ARGS:
    @echo "TODO TASK-026: wire just predict -> uv run python -m alchimiste.cleaner.inference.cli {{ARGS}}"

# Print corpus statistics for the cleaning dataset.
label-stats *ARGS:
    @echo "TODO TASK-011: wire just label-stats -> uv run python -m alchimiste.cleaner.data.label_stats {{ARGS}}"
