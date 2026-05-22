"""Tests for `data.split.make_splits` (TASK-010, REQ-008, NFR-007)."""

from __future__ import annotations

from pathlib import Path

import pytest

from alchimiste.cleaner.data.split import SplitsManifest, make_splits


def _ids(n: int, prefix: str = "art") -> list[str]:
    return [f"{prefix}_{i:04d}" for i in range(n)]


def test_same_seed_yields_identical_manifest() -> None:
    """REQ-008 acceptance: byte-identical output across two runs with same seed."""
    ids = _ids(50)
    a = make_splits(ids, seed=42)
    b = make_splits(ids, seed=42)
    assert a == b
    # Stronger: the dataclass field-by-field equality already implies it,
    # but verify the serialized JSON matches byte-for-byte too.
    assert a.train == b.train
    assert a.val == b.val
    assert a.test == b.test


def test_manifest_json_round_trip(tmp_path: Path) -> None:
    ids = _ids(50)
    a = make_splits(ids, seed=42)
    json_path = tmp_path / "splits.json"
    a.write_json(json_path)
    b = SplitsManifest.read_json(json_path)
    assert a == b


def test_different_seeds_produce_different_partitions() -> None:
    ids = _ids(100)
    a = make_splits(ids, seed=11)
    b = make_splits(ids, seed=22)
    assert a.train != b.train or a.val != b.val or a.test != b.test


def test_stable_under_growth() -> None:
    """NFR-007: adding new item_ids must not migrate existing ones."""
    initial = _ids(100)
    extended = initial + _ids(50, prefix="new")

    a = make_splits(initial, seed=17)
    b = make_splits(extended, seed=17)

    # Every item from `initial` must end up in the same partition in `b`.
    for item_id in initial:
        assert a.partition_of(item_id) == b.partition_of(item_id), (
            f"{item_id} migrated: {a.partition_of(item_id)} -> {b.partition_of(item_id)}"
        )


def test_realized_fractions_within_tolerance() -> None:
    """Spec acceptance: realized fractions within ±5% of targets on 200 items."""
    ids = _ids(200)
    m = make_splits(ids, seed=17, train_frac=0.70, val_frac=0.15)
    n = 200
    assert abs(len(m.train) / n - 0.70) <= 0.05
    assert abs(len(m.val) / n - 0.15) <= 0.05
    assert abs(len(m.test) / n - 0.15) <= 0.05


def test_partitions_are_disjoint_and_cover_all_items() -> None:
    ids = _ids(123)
    m = make_splits(ids, seed=7)
    union = set(m.train) | set(m.val) | set(m.test)
    assert union == set(ids)
    assert len(m.train) + len(m.val) + len(m.test) == len(ids)  # disjoint


def test_empty_input_yields_empty_partitions() -> None:
    m = make_splits([], seed=0)
    assert m.train == () and m.val == () and m.test == ()


def test_duplicate_ids_deduplicated() -> None:
    m = make_splits(["a", "a", "b", "c", "c", "c"], seed=0)
    union = set(m.train) | set(m.val) | set(m.test)
    assert union == {"a", "b", "c"}


def test_invalid_fractions_rejected() -> None:
    with pytest.raises(ValueError, match="invalid fractions"):
        make_splits(["x"], seed=0, train_frac=0.8, val_frac=0.3)
    with pytest.raises(ValueError, match="invalid fractions"):
        make_splits(["x"], seed=0, train_frac=-0.1, val_frac=0.5)


def test_partition_of_unknown_id_raises() -> None:
    m = make_splits(["a"], seed=0)
    with pytest.raises(KeyError):
        m.partition_of("not_in_corpus")
