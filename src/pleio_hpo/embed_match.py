"""Noun-chunk -> nearest HPO term via cosine over a precomputed index.

"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from pleio_hpo.embed import BiomedEmbedder


@dataclass(frozen=True)
class EmbeddingHit:
    start: int
    end: int
    text: str
    hpo_id: str
    score: float
    sentence: str


def default_index_path() -> Path:
    return Path(__file__).parent / "data" / "hpo" / "index.npz"


def load_index(path: Path) -> tuple[np.ndarray, list[str]]:
    """Load the precomputed (n, dim) matrix and the parallel list of HPO IDs."""
    data = np.load(path, allow_pickle=False)
    vectors = data["vectors"].astype(np.float32)
    hpo_ids = [str(x) for x in data["hpo_ids"]]
    return vectors, hpo_ids


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def match(
    chunks: list[tuple[int, int, str, str]],
    embedder: BiomedEmbedder,
    index: tuple[np.ndarray, list[str]],
    threshold: float = 0.8,
    top_k: int = 1,
) -> list[EmbeddingHit]:
    """Match each noun chunk to its nearest HPO term(s) by cosine similarity.

    ``chunks`` are ``(start, end, chunk_text, sentence)`` tuples. Returns at
    most ``top_k`` hits per chunk whose cosine similarity meets ``threshold``.
    """
    if not chunks:
        return []

    index_vectors, hpo_ids = index
    if index_vectors.shape[0] == 0:
        return []

    chunk_texts = [c[2] for c in chunks]
    chunk_vectors = _l2_normalize(embedder.encode(chunk_texts))
    index_norm = _l2_normalize(index_vectors)

    sims = chunk_vectors @ index_norm.T  # (n_chunks, n_index)

    hits: list[EmbeddingHit] = []
    k = max(1, top_k)
    for row, (start, end, text, sentence) in enumerate(chunks):
        scores = sims[row]
        # Indices of the top-k scores, highest first.
        top_idx = np.argsort(scores)[::-1][:k]
        for idx in top_idx:
            score = float(scores[idx])
            if score < threshold:
                continue
            hits.append(
                EmbeddingHit(
                    start=start,
                    end=end,
                    text=text,
                    hpo_id=hpo_ids[idx],
                    score=score,
                    sentence=sentence,
                )
            )
    return hits
