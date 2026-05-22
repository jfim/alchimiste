"""Unit tests for the tokenization cache (loop._tokenize_with_cache).

We don't need a real HF tokenizer here — we patch `_instantiate_tagger`
so the cache logic can be exercised against a deterministic stub. The
goal is to pin:
  * Cache miss writes a pickle keyed on the config + article set.
  * Subsequent identical call hits the cache and doesn't tokenize again.
  * Changing any keyed input (model, max_seq_len, range_units,
    require_nfc, article set) produces a miss.
  * `data.tokenization_cache_dir=""` disables caching entirely.
  * A schema-version mismatch in the payload is ignored, not crashed on.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from alchimiste.cleaner.data.align import TokenizedExample
from alchimiste.cleaner.data.loader import LabeledArticle
from alchimiste.cleaner.training import loop as loop_mod


class _CountingTagger:
    """Records how many `tokenize` calls are made so we can assert
    cache hits don't redo work."""

    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.tokenize_calls = 0

    def tokenize(self, articles, max_seq_len: int):
        self.tokenize_calls += 1
        return [
            TokenizedExample(
                item_id=a.item_id,
                input_ids=(1, 2, 3),
                codepoint_offset_mapping=((0, 0), (0, 1), (1, 2)),
                labels=(-100, 0, 0),
            )
            for a in articles
        ]


@pytest.fixture
def make_cfg(tmp_path: Path):
    def _make(**data_overrides) -> OmegaConf:
        return OmegaConf.create(
            {
                "model": {
                    "name": "stub",
                    "module": "stub_for_tokenization_cache",
                    "hf_model_name": "distilbert-base-uncased",
                },
                "data": {
                    "max_seq_len": 64,
                    "range_units": "codepoint",
                    "require_nfc": True,
                    "tokenization_cache_dir": str(tmp_path / "cache"),
                    **data_overrides,
                },
            }
        )

    return _make


def _articles(n: int = 3) -> list[LabeledArticle]:
    return [
        LabeledArticle(
            item_id=f"a_{i}",
            content_sha256=f"sha_{i}",
            markdown_text=f"text {i}",
            discard_ranges=(),
        )
        for i in range(n)
    ]


def test_cache_miss_then_hit_skips_tokenize_call(make_cfg, tmp_path: Path) -> None:
    cfg = make_cfg()
    tagger = _CountingTagger(cfg.model)

    arts = _articles()
    first = loop_mod._tokenize_with_cache(cfg, arts, tagger=tagger)
    assert tagger.tokenize_calls == len(arts), "first call should tokenize every article"

    second = loop_mod._tokenize_with_cache(cfg, arts, tagger=tagger)
    assert tagger.tokenize_calls == len(arts), (
        "cache hit must not invoke the tagger again"
    )
    assert second == first

    # Cache file must exist and be a single .pkl keyed on the inputs.
    cache_files = list((tmp_path / "cache").glob("*.pkl"))
    assert len(cache_files) == 1


@pytest.mark.parametrize(
    "field, before, after",
    [
        ("max_seq_len", 64, 128),
        ("range_units", "codepoint", "byte"),
        ("require_nfc", True, False),
    ],
)
def test_changing_keyed_input_produces_miss(
    make_cfg, tmp_path: Path, field, before, after
) -> None:
    cfg_a = make_cfg(**{field: before})
    cfg_b = make_cfg(**{field: after})
    arts = _articles()
    tagger_a = _CountingTagger(cfg_a.model)
    tagger_b = _CountingTagger(cfg_b.model)

    loop_mod._tokenize_with_cache(cfg_a, arts, tagger=tagger_a)
    loop_mod._tokenize_with_cache(cfg_b, arts, tagger=tagger_b)
    assert tagger_a.tokenize_calls == len(arts)
    assert tagger_b.tokenize_calls == len(arts), (
        f"changing `data.{field}` must miss the cache"
    )
    assert len(list((tmp_path / "cache").glob("*.pkl"))) == 2


def test_changing_article_set_produces_miss(make_cfg, tmp_path: Path) -> None:
    cfg = make_cfg()
    tagger_a = _CountingTagger(cfg.model)
    tagger_b = _CountingTagger(cfg.model)

    loop_mod._tokenize_with_cache(cfg, _articles(n=3), tagger=tagger_a)
    loop_mod._tokenize_with_cache(cfg, _articles(n=4), tagger=tagger_b)
    assert tagger_a.tokenize_calls == 3
    assert tagger_b.tokenize_calls == 4
    assert len(list((tmp_path / "cache").glob("*.pkl"))) == 2


def test_changing_hf_model_name_produces_miss(make_cfg, tmp_path: Path) -> None:
    cfg_a = make_cfg()
    cfg_b = make_cfg()
    cfg_b.model.hf_model_name = "answerdotai/ModernBERT-base"
    tagger_a = _CountingTagger(cfg_a.model)
    tagger_b = _CountingTagger(cfg_b.model)
    arts = _articles()

    loop_mod._tokenize_with_cache(cfg_a, arts, tagger=tagger_a)
    loop_mod._tokenize_with_cache(cfg_b, arts, tagger=tagger_b)
    assert tagger_b.tokenize_calls == len(arts)


def test_empty_cache_dir_disables_caching(make_cfg, tmp_path: Path) -> None:
    cfg = make_cfg(tokenization_cache_dir="")
    arts = _articles()
    tagger_a = _CountingTagger(cfg.model)
    tagger_b = _CountingTagger(cfg.model)

    loop_mod._tokenize_with_cache(cfg, arts, tagger=tagger_a)
    loop_mod._tokenize_with_cache(cfg, arts, tagger=tagger_b)
    # Both calls retokenize. No cache dir created.
    assert tagger_b.tokenize_calls == len(arts)
    assert not (tmp_path / "cache").exists()


def test_schema_version_mismatch_is_ignored(make_cfg, tmp_path: Path) -> None:
    """A stale cache (different schema) must be re-tokenized and overwritten,
    not crash the run."""
    cfg = make_cfg()
    arts = _articles()
    tagger = _CountingTagger(cfg.model)
    loop_mod._tokenize_with_cache(cfg, arts, tagger=tagger)
    assert tagger.tokenize_calls == len(arts)

    # Corrupt the cache: replace the payload's schema marker.
    cache_files = list((tmp_path / "cache").glob("*.pkl"))
    assert len(cache_files) == 1
    cf = cache_files[0]
    with cf.open("rb") as fh:
        payload = pickle.load(fh)
    payload["schema"] = "v999_stale"
    with cf.open("wb") as fh:
        pickle.dump(payload, fh)

    tagger2 = _CountingTagger(cfg.model)
    loop_mod._tokenize_with_cache(cfg, arts, tagger=tagger2)
    # Stale cache → fell through to retokenize.
    assert tagger2.tokenize_calls == len(arts)
    # And the overwritten cache should now be valid for next call.
    tagger3 = _CountingTagger(cfg.model)
    loop_mod._tokenize_with_cache(cfg, arts, tagger=tagger3)
    assert tagger3.tokenize_calls == 0, "post-overwrite cache should hit"
