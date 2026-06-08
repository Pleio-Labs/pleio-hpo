"""HPO ontology loader: parse hp.obo, expose terms and the parent DAG.

"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import fastobo


@dataclass
class HPOTerm:
    id: str
    label: str
    synonyms: list[str] = field(default_factory=list)
    parents: list[str] = field(default_factory=list)
    definition: str = ""
    is_obsolete: bool = False


def default_hpo_path() -> Path:
    return Path(__file__).parent / "data" / "hpo" / "hp.obo"


def bundled_hpo_version() -> str:
    """Read the bundled HPO release tag (e.g. ``v2026-02-16``)."""
    version_path = Path(__file__).parent / "data" / "hpo" / "HPO_VERSION"
    return version_path.read_text().strip() if version_path.exists() else "unknown"


_HP_RE = re.compile(r"^HP:\d{7}$")
_ALT_ID_RE = re.compile(r"^HP:\d{7}$")


def load_terms(path: Path) -> list[HPOTerm]:
    """Parse an OBO file and return non-obsolete HPO terms."""
    doc = fastobo.load(str(path))
    terms: list[HPOTerm] = []

    for frame in doc:
        if not isinstance(frame, fastobo.term.TermFrame):
            continue

        term_id = str(frame.id)
        if not _HP_RE.match(term_id):
            continue

        label = ""
        synonyms: list[str] = []
        parents: list[str] = []
        definition = ""
        is_obsolete = False

        for clause in frame:
            tag = clause.__class__.__name__

            if tag == "NameClause":
                label = str(clause.name)
            elif tag == "DefClause":
                definition = str(clause.definition)
            elif tag == "IsObsoleteClause":
                is_obsolete = clause.obsolete
            elif tag == "IsAClause":
                parent_id = str(clause.term)
                if _HP_RE.match(parent_id):
                    parents.append(parent_id)
            elif tag == "SynonymClause":
                synonyms.append(str(clause.synonym.desc))

        if is_obsolete:
            continue

        terms.append(
            HPOTerm(
                id=term_id,
                label=label,
                synonyms=synonyms,
                parents=parents,
                definition=definition,
                is_obsolete=False,
            )
        )

    return terms


def build_parent_map(terms: list[HPOTerm]) -> dict[str, list[str]]:
    """Return {child_id: [parent_id, ...]} from a list of HPOTerms."""
    return {t.id: list(t.parents) for t in terms if t.parents}


PHENOTYPIC_ABNORMALITY_ROOT = "HP:0000118"
MODE_OF_INHERITANCE_ROOT = "HP:0000005"
ONSET_ROOT = "HP:0003674"
# Default candidate vocabulary = phenotypic abnormalities PLUS inheritance and onset.
# Inheritance/onset are not "abnormalities" per se, but clinicians and reference
# corpora (e.g. GSC+) record them ("autosomal dominant", "infantile onset") and
# their surface forms almost never fire spuriously — unlike the Clinical-Modifier
# branch (HP:0012823: "bilateral", "mild", "severe"), which stays excluded.
# (Decision 2026-06-04, recall diagnostic on GSC+.)
DEFAULT_CANDIDATE_ROOTS = (
    PHENOTYPIC_ABNORMALITY_ROOT,
    MODE_OF_INHERITANCE_ROOT,
    ONSET_ROOT,
)
# Sub-branches pruned from the candidate vocabulary. "Inheritance qualifier"
# (HP:0034335) holds abstract genetics-mechanism descriptors — uniparental disomy,
# imprinting, penetrance/mosaicism qualifiers — that are not clinical phenotypes a
# note would record; they were the entire false-positive source when the
# inheritance branch was re-included (verified on GSC+). We keep the inheritance
# *patterns* (Mendelian, sporadic, ...) and drop the qualifiers.
DEFAULT_CANDIDATE_EXCLUDE = ("HP:0034335",)


def subtree_ids(terms: list[HPOTerm], roots: tuple[str, ...]) -> set[str]:
    """All term IDs strictly *below* any of ``roots`` (BFS over the parent graph).

    The ``roots`` themselves are abstract hub nodes ("Phenotypic abnormality",
    "Mode of inheritance", "Onset") that are never valid annotations, so they are
    excluded — only their descendant leaves/terms are kept.
    """
    children: dict[str, set[str]] = {}
    for t in terms:
        for parent in t.parents:
            children.setdefault(parent, set()).add(t.id)
    keep: set[str] = set()
    frontier = set(roots)
    while frontier:
        nxt: set[str] = set()
        for node in frontier:
            nxt |= children.get(node, set())
        nxt -= keep
        keep |= nxt
        frontier = nxt
    return keep


def phenotypic_abnormality_ids(
    terms: list[HPOTerm],
    roots: tuple[str, ...] = DEFAULT_CANDIDATE_ROOTS,
    exclude: tuple[str, ...] = DEFAULT_CANDIDATE_EXCLUDE,
) -> set[str]:
    """Candidate-vocabulary IDs: the ``roots`` subtrees minus the ``exclude`` subtrees.

    Defaults to phenotypic abnormalities + inheritance + onset
    (:data:`DEFAULT_CANDIDATE_ROOTS`), minus the abstract Inheritance-qualifier
    branch (:data:`DEFAULT_CANDIDATE_EXCLUDE`). Restricting the candidate
    vocabulary this way is an a-priori design choice, not tuning — it keeps the
    false-positive Clinical-Modifier / Frequency / Past-history / inheritance-
    qualifier branches out. The full ontology is still used for scoring, valid-ID
    filtering, and ancestor maps.
    """
    return subtree_ids(terms, roots) - subtree_ids(terms, exclude)
