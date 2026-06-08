"""Unit tests for embed.py.

These download SapBERT (~440 MB) on first run.
"""

from __future__ import annotations

import numpy as np
import pytest

from pleio_hpo.embed import BiomedEmbedder


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def test_distinct_phrases_same_dim_different_content() -> None:
    emb = BiomedEmbedder()
    vecs = emb.encode(["macrocephaly", "seizures"])
    assert vecs.shape[0] == 2
    assert vecs[0].shape == vecs[1].shape
    assert not np.allclose(vecs[0], vecs[1])


def test_identical_strings_cosine_one() -> None:
    emb = BiomedEmbedder()
    vecs = emb.encode(["macrocephaly", "macrocephaly"])
    assert _cosine(vecs[0], vecs[1]) == pytest.approx(1.0, abs=1e-5)
