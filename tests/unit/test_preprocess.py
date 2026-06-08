"""Unit tests for preprocess.py."""

from __future__ import annotations

from pleio_hpo.preprocess import noun_chunks, sentences


def test_sentence_splitting_offsets() -> None:
    text = "The patient has macrocephaly. She also has hypotonia."
    sents = sentences(text)
    assert len(sents) == 2
    s0, e0, t0 = sents[0]
    s1, e1, t1 = sents[1]
    assert t0 == text[s0:e0]
    assert t1 == text[s1:e1]
    assert "macrocephaly" in t0
    assert "hypotonia" in t1


def test_noun_chunks_contain_phenotype() -> None:
    text = "The patient has global developmental delay."
    chunks = noun_chunks(text)
    texts = [c[2].lower() for c in chunks]
    assert any("global developmental delay" in t for t in texts)
