"""Tests for `models.base.TokenTagger` Protocol (TASK-012)."""

from __future__ import annotations

from pathlib import Path

from omegaconf import DictConfig

from alchimiste.cleaner.data.align import TokenizedExample
from alchimiste.cleaner.models.base import TokenTagger, TrainingCallbacks


class _StubTagger:
    """Minimal `TokenTagger` impl — exists to prove the Protocol is satisfiable
    by a plain class without inheritance (structural typing)."""

    def fit(
        self,
        train: list[TokenizedExample],
        val: list[TokenizedExample],
        cfg: DictConfig,
        callbacks: TrainingCallbacks,
    ) -> None:
        callbacks.on_batch_end(0, 0.0)
        callbacks.on_epoch_end(0, {"loss": 0.0})

    def predict_token_probs(self, examples: list[TokenizedExample]) -> list[list[float]]:
        return [[0.0] * len(e.input_ids) for e in examples]

    def save(self, dst: Path) -> None:
        dst.mkdir(parents=True, exist_ok=True)
        (dst / "marker").write_text("stub", encoding="utf-8")

    @classmethod
    def load(cls, src: Path) -> _StubTagger:
        assert (src / "marker").read_text(encoding="utf-8") == "stub"
        return cls()


def test_stub_satisfies_protocol_via_isinstance() -> None:
    tagger = _StubTagger()
    assert isinstance(tagger, TokenTagger)


def test_callbacks_defaults_are_noops() -> None:
    cb = TrainingCallbacks()
    # Calling the defaults shouldn't raise; they're plain lambdas that
    # return None.
    cb.on_batch_end(1, 0.5)
    cb.on_epoch_end(2, {"f1": 0.0})


def test_callbacks_record_calls(tmp_path: Path) -> None:
    batch_calls: list[tuple[int, float]] = []
    epoch_calls: list[tuple[int, dict[str, float]]] = []
    cb = TrainingCallbacks(
        on_batch_end=lambda step, loss: batch_calls.append((step, loss)),
        on_epoch_end=lambda epoch, metrics: epoch_calls.append((epoch, metrics)),
    )
    cfg = DictConfig({"placeholder": True})
    _StubTagger().fit(train=[], val=[], cfg=cfg, callbacks=cb)
    assert batch_calls == [(0, 0.0)]
    assert epoch_calls == [(0, {"loss": 0.0})]


def test_save_load_round_trip(tmp_path: Path) -> None:
    tagger = _StubTagger()
    tagger.save(tmp_path / "art")
    restored = _StubTagger.load(tmp_path / "art")
    assert isinstance(restored, TokenTagger)


def test_predict_returns_per_token_probs() -> None:
    ex = TokenizedExample(
        item_id="t",
        input_ids=(1, 2, 3),
        codepoint_offset_mapping=((0, 0), (0, 1), (1, 2)),
        labels=(-100, 0, 0),
    )
    probs = _StubTagger().predict_token_probs([ex])
    assert probs == [[0.0, 0.0, 0.0]]
