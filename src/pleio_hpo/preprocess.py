"""Text normalization, sentence splitting, and noun-chunk extraction (spaCy).

"""

from __future__ import annotations

_nlp = None


def _get_nlp():
    global _nlp
    if _nlp is None:
        import spacy

        try:
            _nlp = spacy.load("en_core_web_sm")
        except OSError as exc:
            raise OSError(
                "spaCy model 'en_core_web_sm' is not installed. Fetch it with "
                "`pleio-hpo download` (or `python -m spacy download en_core_web_sm`)."
            ) from exc
    return _nlp


def sentences(text: str) -> list[tuple[int, int, str]]:
    """Return list of (start, end, text) tuples — one per sentence."""
    nlp = _get_nlp()
    doc = nlp(text)
    return [(sent.start_char, sent.end_char, sent.text) for sent in doc.sents]


def noun_chunks(text: str) -> list[tuple[int, int, str]]:
    """Return list of (start, end, text) noun-chunk tuples."""
    nlp = _get_nlp()
    doc = nlp(text)
    return [(chunk.start_char, chunk.end_char, chunk.text) for chunk in doc.noun_chunks]
