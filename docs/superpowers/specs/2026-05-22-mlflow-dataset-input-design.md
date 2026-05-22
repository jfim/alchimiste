# MLflow Dataset Input — Design

**Date:** 2026-05-22
**Status:** Draft

## Goal

Surface each training run's dataset as a first-class MLflow **Dataset** (visible in the UI's Datasets column) rather than only as run tags. The dataset is identified by its oxen commit, so logging a pointer to that commit is sufficient — no data needs to be materialized to MLflow.

## Naming

Format: `{prefix}-{short_hash}-n-{article_count}` with optional `-dirty` suffix.

Example: `cleaning-178fc8-n-193`, `cleaning-178fc8-n-193-dirty`.

- **prefix**: `Path(cfg.data.oxen_dir).name` — e.g. `cleaning` for `alchimiste-data/cleaning`. Auto-renames as we add other oxen subtrees (e.g. a future `ocr/`).
- **short_hash**: first 6 chars of `OxenMeta.commit_hash`.
- **article_count (`n`)**: number of entries in `<oxen_dir>/blobs/`, i.e. distinct articles in the dataset version.
- **`-dirty` suffix**: appended when `OxenMeta.dirty=True`, so dirty runs don't share a dataset identity with the clean commit in the UI.

## MLflow API

Use `mlflow.data.meta_dataset.MetaDataset` — MLflow's pointer-only dataset type, intended for cases where the dataset's rows aren't loaded into the run process. Backed by a small custom `DatasetSource` subclass whose URI is `oxen://{prefix}@{commit_hash}`.

This is the right primitive because:

- We don't load `rows.parquet` just to log identity.
- The URI surfaces in the UI when the dataset is expanded.
- MLflow renders it in the Datasets column with the chosen `name`.

The dataset is logged via `mlflow.log_input(dataset)` inside `start_run()`, immediately after `set_tags(...)`.

## Integration Point

All changes live in `src/alchimiste/cleaner/training/mlflow_io.py`:

1. **New `OxenDatasetSource`** (subclass of `mlflow.data.dataset_source.DatasetSource`): wraps the oxen URI; implements the four required methods (`_get_source_type`, `load`, `_to_dict`, `_from_dict`). `load` raises `NotImplementedError` — this is a pointer, not loadable through MLflow.
2. **New `_build_dataset(oxen_meta, oxen_dir) -> MetaDataset`**: assembles the source + name per the rules above.
3. **`start_run()` change**: after `mlflow.set_tags(tags)`, if `oxen_meta is not None`, call `mlflow.log_input(_build_dataset(oxen_meta, cfg.data.oxen_dir))`.

Existing oxen-identity tags (`alchimiste.dataset.oxen_commit`, `alchimiste.dataset.dirty`, `alchimiste.dataset.oxen_dir`) remain — they're still useful for free-text filter queries in the MLflow UI.

## Edge Cases

| Case | Behavior |
|---|---|
| `oxen_meta is None` (e.g. fixture-only tests) | Skip `log_input` entirely. |
| `<oxen_dir>/blobs/` missing | Fall back to `rows.parquet` row count if present; else `n=?`. Log a warning, don't crash the run. |
| `<oxen_dir>` itself missing | `_build_dataset` is not reached because `oxen_meta` would already be `None`. |
| Counting `blobs/` | `sum(1 for _ in path.iterdir())` — one syscall per entry; trivial at the current ~200-file scale and fine at 10× that. |

## Tests

New file: `tests/cleaner/test_mlflow_dataset.py`.

- `_build_dataset` produces the expected name for: clean tree, dirty tree, missing `blobs/` with `rows.parquet` fallback, both missing.
- `start_run` calls `mlflow.log_input` exactly once with the constructed dataset when `oxen_meta` is provided.
- `start_run` does not call `log_input` when `oxen_meta is None`.
- `OxenDatasetSource` round-trips through `_to_dict` / `_from_dict`.

Existing oxen-meta tag tests must stay green (no tag removals).

## Out of Scope

- Changes to the registered pyfunc model or any other MLflow surface.
- Loading dataset rows into MLflow (the data lives in oxen; that's the source of truth).
- Backfilling historical runs.
- Renaming or removing existing dataset tags.
