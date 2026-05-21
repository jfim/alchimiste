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
