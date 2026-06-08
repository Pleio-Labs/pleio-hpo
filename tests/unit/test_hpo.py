"""Unit tests for hpo.py."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from pleio_hpo.hpo import build_parent_map, load_terms

OBO_FIXTURE = textwrap.dedent("""\
    format-version: 1.2

    [Term]
    id: HP:0000001
    name: All
    comment: Root of HPO.

    [Term]
    id: HP:0000002
    name: Child term
    def: "A child of the root." [HPO:test]
    synonym: "little child" EXACT []
    synonym: "small child" RELATED []
    is_a: HP:0000001 ! All

    [Term]
    id: HP:0000003
    name: Grandchild term
    is_a: HP:0000002 ! Child term

    [Term]
    id: HP:9999999
    name: Old thing
    is_obsolete: true
""")


@pytest.fixture()
def obo_path(tmp_path: Path) -> Path:
    p = tmp_path / "test.obo"
    p.write_text(OBO_FIXTURE)
    return p


def test_load_basic_ids_and_labels(obo_path: Path) -> None:
    terms = load_terms(obo_path)
    ids = {t.id for t in terms}
    # 3 non-obsolete HP terms
    assert ids == {"HP:0000001", "HP:0000002", "HP:0000003"}
    by_id = {t.id: t for t in terms}
    assert by_id["HP:0000002"].label == "Child term"


def test_parent_map(obo_path: Path) -> None:
    terms = load_terms(obo_path)
    pm = build_parent_map(terms)
    assert pm["HP:0000002"] == ["HP:0000001"]
    assert pm["HP:0000003"] == ["HP:0000002"]
    # Root has no parents
    assert "HP:0000001" not in pm


def test_obsolete_filtered(obo_path: Path) -> None:
    terms = load_terms(obo_path)
    assert all(t.id != "HP:9999999" for t in terms)


def test_synonyms_collected(obo_path: Path) -> None:
    terms = load_terms(obo_path)
    child = next(t for t in terms if t.id == "HP:0000002")
    assert "little child" in child.synonyms
    assert "small child" in child.synonyms


def test_definition_parsed(obo_path: Path) -> None:
    terms = load_terms(obo_path)
    child = next(t for t in terms if t.id == "HP:0000002")
    assert child.definition == "A child of the root."
    root = next(t for t in terms if t.id == "HP:0000001")
    assert root.definition == ""
