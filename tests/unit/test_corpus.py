"""Unit tests for eval/corpus.py.

Uses tiny fixtures in tests/fixtures/corpora/ plus the real committed consult
eval for the count check in the [VERIFY] gate.
"""

from __future__ import annotations

from pathlib import Path

from pleio_hpo.eval.corpus import (
    load_biocreative_viii_train,
    load_consult,
    load_gsc_plus,
)

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "corpora"


def test_load_gsc_plus_three_docs() -> None:
    docs, skipped = load_gsc_plus(FIXTURES / "gsc_plus")
    assert len(docs) == 3
    assert skipped == 0
    by_id = {d.doc_id: d for d in docs}

    doc1 = by_id["doc1"]
    assert doc1.hpo_ids == {"HP:0000256", "HP:0001252"}
    macro = next(a for a in doc1.annotations if a.hpo_id == "HP:0000256")
    assert (macro.start, macro.end, macro.text) == (12, 24, "macrocephaly")

    # HP_ -> HP: conversion and multi-word span
    assert by_id["doc3"].annotations[0].text == "global developmental delay"
    assert by_id["doc3"].annotations[0].hpo_id == "HP:0001263"


def test_load_gsc_plus_filters_obsolete_ids() -> None:
    # Only HP:0000256 is "valid" -> the other gold annotations are skipped.
    docs, skipped = load_gsc_plus(FIXTURES / "gsc_plus", valid_ids={"HP:0000256"})
    kept = {a.hpo_id for d in docs for a in d.annotations}
    assert kept == {"HP:0000256"}
    assert skipped > 0


def test_load_bc8_grouping_and_spans() -> None:
    docs = load_biocreative_viii_train(FIXTURES / "bc8")
    by_id = {d.doc_id: d for d in docs}
    assert set(by_id) == {"s1", "s2", "s3"}

    # normal row
    assert by_id["s1"].annotations[0].text == "High arched palate"
    # NA row -> no annotations
    assert by_id["s2"].annotations == []
    # discontinuous span -> longest sub-span (20-25), negated flag carried
    s3 = by_id["s3"].annotations[0]
    assert (s3.start, s3.end, s3.text) == (20, 25, "cleft")
    assert s3.negated is True


def test_load_consult_fixture() -> None:
    docs = load_consult(FIXTURES / "consult" / "three.jsonl")
    assert len(docs) == 3
    assert docs[0].doc_id == "c1"
    assert docs[0].annotations[0].hpo_id == "HP:0000256"
