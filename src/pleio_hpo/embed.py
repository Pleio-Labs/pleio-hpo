"""Biomedical sentence-transformer wrapper (SapBERT / BioLORD).

"""

from __future__ import annotations

import numpy as np

DEFAULT_MODEL = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
# Pinned for reproducibility.
DEFAULT_REVISION = "090663c3ae57bf35ffe4d0d468a2a88d03051a4d"


class BiomedEmbedder:
    """Lazy wrapper around a sentence-transformers model.

    The model (~440 MB for SapBERT) is downloaded and loaded on first
    ``encode`` call, not at construction time. When the default model is used,
    its HuggingFace revision is pinned so weights match the committed index.
    """

    def __init__(self, model_name: str = DEFAULT_MODEL, revision: str | None = None) -> None:
        self._model_name = model_name
        if revision is not None:
            self._revision: str | None = revision
        elif model_name == DEFAULT_MODEL:
            self._revision = DEFAULT_REVISION
        else:
            self._revision = None
        self._model = None

    @property
    def model_name(self) -> str:
        return self._model_name

    def _ensure_loaded(self) -> None:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self._model_name, revision=self._revision)

    def encode(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        """Encode *texts* into a ``(n, dim)`` float32 array."""
        self._ensure_loaded()
        assert self._model is not None
        vectors = self._model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=np.float32)
