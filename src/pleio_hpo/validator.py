"""PubMedBERT cross-encoder validator (inference only).

The validator is a *two-segment* cross-encoder, not a triple. Per the resolved
spec, each candidate is encoded as::

    left  = sentence with the mention wrapped in "[E]" ... "[/E]"
    right = f"{hpo_label} : {hpo_def}"
    tokenizer(left, right, truncation=True, max_length=256)

"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

MARKER_OPEN = "[E]"
MARKER_CLOSE = "[/E]"
MAX_LENGTH = 256


def default_validator_path() -> Path:
    return Path(__file__).parent / "data" / "validator" / "v1.0"


@dataclass(frozen=True)
class ValidatorScore:
    keep: bool
    score: float  # positive-class probability, in [0, 1]


@dataclass(frozen=True)
class ValidatorCandidate:
    """One (mention-in-sentence, HPO term) pair to validate.

    ``mention_start``/``mention_end`` are offsets into ``sentence`` (not the
    whole document), matching the training-time ``build_pair`` recipe.
    """

    sentence: str
    mention_start: int
    mention_end: int
    hpo_label: str
    hpo_def: str


def build_pair(candidate: ValidatorCandidate) -> tuple[str, str]:
    """Construct the exact (left, right) segment pair used at training time."""
    s = candidate.sentence
    a, b = candidate.mention_start, candidate.mention_end
    left = s[:a] + MARKER_OPEN + s[a:b] + MARKER_CLOSE + s[b:]
    right = f"{candidate.hpo_label} : {candidate.hpo_def}"
    return left, right


class Validator:
    def __init__(
        self, model_path: Path, threshold: float = 0.5, batch_size: int = 32
    ) -> None:
        self._model_path = model_path
        self._threshold = threshold
        self._batch_size = batch_size
        self._tokenizer: Any = None
        self._model: Any = None

    def _ensure_loaded(self) -> None:
        if self._model is None:
            from transformers import (
                AutoModelForSequenceClassification,
                AutoTokenizer,
            )

            self._tokenizer = AutoTokenizer.from_pretrained(str(self._model_path))
            self._model = AutoModelForSequenceClassification.from_pretrained(
                str(self._model_path)
            )
            self._model.eval()

    def score(self, candidates: list[ValidatorCandidate]) -> list[ValidatorScore]:
        """Batch inference. Returns one ValidatorScore per candidate."""
        if not candidates:
            return []

        self._ensure_loaded()
        assert self._tokenizer is not None and self._model is not None

        import torch

        results: list[ValidatorScore] = []
        for start in range(0, len(candidates), self._batch_size):
            batch = candidates[start : start + self._batch_size]
            lefts, rights = zip(*(build_pair(c) for c in batch), strict=True)
            encoded = self._tokenizer(
                list(lefts),
                list(rights),
                truncation=True,
                max_length=MAX_LENGTH,
                padding=True,
                return_tensors="pt",
            )
            with torch.no_grad():
                logits = self._model(**encoded).logits
            for p in _positive_probabilities(logits):
                results.append(
                    ValidatorScore(keep=p >= self._threshold, score=p)
                )
        return results


def _positive_probabilities(logits) -> list[float]:
    """Map model logits to a positive-class probability per row.

    Supports both single-logit (binary) and two-logit (softmax) heads.
    """
    import torch

    if logits.ndim == 2 and logits.shape[1] == 1:
        probs = torch.sigmoid(logits.squeeze(-1))
    elif logits.ndim == 2 and logits.shape[1] >= 2:
        probs = torch.softmax(logits, dim=-1)[:, 1]
    else:
        probs = torch.sigmoid(logits.reshape(-1))
    return [float(x) for x in probs.tolist()]
