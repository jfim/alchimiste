"""Verify `_seed_all_rngs` actually pins every RNG that affects training.

Before this seeding was added, only `make_splits` consumed `cfg.seed`;
PyTorch, NumPy, and Python `random` all came from un-seeded global state,
so freshly-init'd classifier head weights / DataLoader shuffle order /
dropout masks varied per process. That caused F1 swings far larger than
any hyperparameter effect we were trying to measure.
"""

from __future__ import annotations

import random

import numpy as np
import torch

from alchimiste.cleaner.training.loop import _seed_all_rngs


def _draw_samples() -> tuple[float, float, list[float]]:
    """Pull one number from each RNG. Order matters."""
    return (
        random.random(),
        float(np.random.rand()),
        torch.rand(4).tolist(),
    )


def test_same_seed_reproduces_random_torch_and_numpy_draws() -> None:
    _seed_all_rngs(17)
    a = _draw_samples()
    _seed_all_rngs(17)
    b = _draw_samples()
    assert a == b, f"same-seed RNG draws should match exactly; got {a!r} vs {b!r}"


def test_different_seeds_produce_different_draws() -> None:
    _seed_all_rngs(17)
    a = _draw_samples()
    _seed_all_rngs(42)
    b = _draw_samples()
    assert a != b, f"different seeds should differ in at least one RNG; got {a!r} == {b!r}"


def test_torch_linear_init_is_deterministic_under_seed() -> None:
    """A freshly-init'd linear layer (analog of the classifier head) lands
    on identical weights when seeded identically — and the *pre-seeding*
    behavior would have left it varying per process."""
    _seed_all_rngs(17)
    head_a = torch.nn.Linear(8, 2)
    weights_a = head_a.weight.detach().clone()

    _seed_all_rngs(17)
    head_b = torch.nn.Linear(8, 2)
    weights_b = head_b.weight.detach().clone()

    assert torch.equal(weights_a, weights_b), (
        "Classifier-head init must be reproducible under the same seed — "
        "this was the dominant source of cross-run variance before seeding "
        "was added to the training loop."
    )
