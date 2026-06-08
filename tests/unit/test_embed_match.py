"""Unit tests for embed_match.py.

Builds a tiny in-memory index from a handful of HPO terms using the real
SapBERT embedder (downloaded once, shared via the module-scoped fixtures).
"""

from __future__ import annotations

import numpy as np
import pytest

from pleio_hpo.embed import BiomedEmbedder
from pleio_hpo.embed_match import match

# A small HPO-like vocabulary: (hpo_id, surface_string).
VOCAB = [
    ("HP:0000256", "Macrocephaly"),
    ("HP:0001250", "Seizure"),
    ("HP:0001263", "Global developmental delay"),
    ("HP:0000545", "Myopia"),
    ("HP:0001252", "Hypotonia"),
]


@pytest.fixture(scope="module")
def embedder() -> BiomedEmbedder:
    return BiomedEmbedder()


@pytest.fixture(scope="module")
def index(embedder: BiomedEmbedder) -> tuple[np.ndarray, list[str]]:
    vectors = embedder.encode([s for _, s in VOCAB])
    return vectors, [hid for hid, _ in VOCAB]


def test_semantic_match_above_threshold(embedder, index) -> None:
    # "big head" is semantically close to "Macrocephaly".
    chunks = [(0, 8, "big head", "She has a big head.")]
    hits = match(chunks, embedder, index, threshold=0.5)
    assert len(hits) == 1
    assert hits[0].hpo_id == "HP:0000256"
    assert hits[0].score >= 0.5
    assert hits[0].sentence == "She has a big head."


def test_non_phenotype_chunk_no_hit(embedder, index) -> None:
    # "the patient" should not be near any phenotype at a high threshold.
    chunks = [(0, 11, "the patient", "The patient was seen today.")]
    hits = match(chunks, embedder, index, threshold=0.8)
    assert hits == []
