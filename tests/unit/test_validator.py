"""Unit tests for validator.py.

Uses a mock model/tokenizer with canned logits — no trained checkpoint here
(real-checkpoint integration is verified in Task 5.3).
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from pleio_hpo.validator import (
    Validator,
    ValidatorCandidate,
    build_pair,
)


def _logit(p: float) -> float:
    return math.log(p / (1 - p))


class _FakeTokenizer:
    def __call__(self, lefts, rights, **kwargs):
        return {"batch_len": len(lefts)}


class _FakeOutput:
    def __init__(self, logits) -> None:
        self.logits = logits


class _FakeModel:
    """Pops the next ``batch_len`` preset positive-class probs per call."""

    def __init__(self, probs: list[float]) -> None:
        self._probs = list(probs)
        self.call_batch_sizes: list[int] = []

    def eval(self) -> None:
        pass

    def __call__(self, **encoded):
        import torch

        n = encoded["batch_len"]
        self.call_batch_sizes.append(n)
        chunk = [self._probs.pop(0) for _ in range(n)]
        # Two-logit softmax head: [0, logit(q)] -> softmax gives [1-q, q].
        logits = torch.tensor([[0.0, _logit(q)] for q in chunk])
        return _FakeOutput(logits)


def _mock_validator(probs: list[float], threshold: float = 0.5, batch_size: int = 32) -> Validator:
    v = Validator(Path("/nonexistent"), threshold=threshold, batch_size=batch_size)
    v._tokenizer = _FakeTokenizer()
    v._model = _FakeModel(probs)
    return v


def _cand(label: str = "Macrocephaly") -> ValidatorCandidate:
    return ValidatorCandidate(
        sentence="Patient has macrocephaly today.",
        mention_start=12,
        mention_end=24,
        hpo_label=label,
        hpo_def="An abnormally large head.",
    )


def test_build_pair_wraps_mention_and_joins_def() -> None:
    left, right = build_pair(_cand())
    assert left == "Patient has [E]macrocephaly[/E] today."
    assert right == "Macrocephaly : An abnormally large head."


def test_threshold_keep_reject() -> None:
    v = _mock_validator([0.9, 0.3, 0.6], threshold=0.5)
    scores = v.score([_cand(), _cand(), _cand()])
    assert [s.keep for s in scores] == [True, False, True]
    assert [round(s.score, 3) for s in scores] == [0.9, 0.3, 0.6]


def test_batching_splits_and_preserves_order() -> None:
    v = _mock_validator([0.9, 0.1, 0.8], threshold=0.5, batch_size=2)
    scores = v.score([_cand(), _cand(), _cand()])
    # batch_size=2 over 3 candidates -> calls of size 2 then 1
    assert v._model.call_batch_sizes == [2, 1]
    assert [s.keep for s in scores] == [True, False, True]


def test_empty_candidates() -> None:
    v = _mock_validator([])
    assert v.score([]) == []


def test_threshold_boundary_is_inclusive() -> None:
    v = _mock_validator([0.5], threshold=0.5)
    assert v.score([_cand()])[0].keep is True


@pytest.mark.parametrize("p", [0.05, 0.95])
def test_score_in_unit_interval(p: float) -> None:
    v = _mock_validator([p])
    s = v.score([_cand()])[0]
    assert 0.0 <= s.score <= 1.0
