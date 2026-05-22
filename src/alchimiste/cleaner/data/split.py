"""Deterministic train/val/test splitter keyed on `item_id` (REQ-008, NFR-007).

The split is computed by hashing `(seed, item_id)` to a stable float in
[0, 1) and bucketing into partitions by cumulative fraction. Two important
properties fall out for free:

* **Reproducible.** Same `(seed, item_ids)` produce byte-identical manifests
  across runs and machines (REQ-008 acceptance criterion).
* **Stable under growth.** Adding new `item_id`s to the corpus leaves every
  previously-assigned item in its original partition — there's no global
  re-shuffle, so an item never migrates between train and test as the
  active labeling loop adds rows (NFR-007).

We use `hashlib.sha256` rather than the built-in `hash()` because Python's
hash randomization makes `hash()` non-stable across processes.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

Partition = Literal["train", "val", "test"]


@dataclass(frozen=True)
class SplitsManifest:
    """Per-partition `item_id` lists plus the seed used to produce them."""

    seed: int
    train_frac: float
    val_frac: float
    test_frac: float
    train: tuple[str, ...]
    val: tuple[str, ...]
    test: tuple[str, ...]

    def partition_of(self, item_id: str) -> Partition:
        if item_id in self.train:
            return "train"
        if item_id in self.val:
            return "val"
        if item_id in self.test:
            return "test"
        raise KeyError(item_id)

    def write_json(self, path: Path) -> None:
        path.write_text(
            json.dumps(asdict(self), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def read_json(cls, path: Path) -> SplitsManifest:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            seed=raw["seed"],
            train_frac=raw["train_frac"],
            val_frac=raw["val_frac"],
            test_frac=raw["test_frac"],
            train=tuple(raw["train"]),
            val=tuple(raw["val"]),
            test=tuple(raw["test"]),
        )


def make_splits(
    item_ids: Iterable[str],
    *,
    seed: int,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
) -> SplitsManifest:
    """Deterministically assign each `item_id` to a partition.

    `test_frac` is implicit as `1 - train_frac - val_frac`. Raises
    `ValueError` if the fractions are negative or sum to more than 1.
    """
    if train_frac < 0 or val_frac < 0 or train_frac + val_frac > 1.0:
        raise ValueError(
            f"invalid fractions: train={train_frac}, val={val_frac}; "
            "must be non-negative and sum to <= 1.0"
        )
    test_frac = 1.0 - train_frac - val_frac

    # Dedup + sort for stable iteration order. Sorting is what makes the
    # output canonical for byte-identical manifest comparison.
    unique_sorted = sorted(set(item_ids))

    train: list[str] = []
    val: list[str] = []
    test: list[str] = []

    for item_id in unique_sorted:
        h = _hash_to_unit_interval(seed=seed, item_id=item_id)
        if h < train_frac:
            train.append(item_id)
        elif h < train_frac + val_frac:
            val.append(item_id)
        else:
            test.append(item_id)

    return SplitsManifest(
        seed=seed,
        train_frac=train_frac,
        val_frac=val_frac,
        test_frac=test_frac,
        train=tuple(train),
        val=tuple(val),
        test=tuple(test),
    )


def _hash_to_unit_interval(*, seed: int, item_id: str) -> float:
    """Map `(seed, item_id)` to a stable float in [0, 1).

    Uses sha256 of `<seed>|<item_id>` (the pipe is a delimiter that can't
    appear in a typical `item_id`; even if it did, sha256's preimage
    resistance keeps collisions astronomically rare). Takes the top 64
    bits and divides by 2**64.
    """
    payload = f"{seed}|{item_id}".encode()
    digest = hashlib.sha256(payload).digest()
    top_64 = int.from_bytes(digest[:8], byteorder="big", signed=False)
    return top_64 / (1 << 64)
