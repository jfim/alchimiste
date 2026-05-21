# alchimiste

Two ML models for use with [alembic](../alembic):

1. **DOM node extractor** — readability-style model that selects the single DOM node containing the primary content of a page.
2. **Text cleaner** — removes non-relevant text items (boilerplate, nav, ads, repeated cruft) from extracted content.

## Status

Early scaffolding. Model architectures and training pipelines are not yet designed.

## Requirements

- Python ≥ 3.12
- [uv](https://github.com/astral-sh/uv) for dependency management
- [just](https://github.com/casey/just) for task running

## Setup

```sh
just sync
```

## Common tasks

```sh
just            # list available recipes
just lint       # ruff check
just fmt        # ruff format
just test       # pytest
just check      # lint + format check + tests
```

## Layout

```
src/alchimiste/   # package source
tests/            # pytest tests
```

## License

Apache 2.0 — see [LICENSE](LICENSE).
