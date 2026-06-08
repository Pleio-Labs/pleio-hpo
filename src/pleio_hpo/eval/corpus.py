"""Corpus loaders: GSC+, BioCreative VIII Track 3, and the consult eval.


Obsolete-HPO-ID filtering (GSC+ §1.1) is deferred until hpo.py exists;
for now the loaders return every annotation.
"""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path

from pleio_hpo.eval.metrics import Annotation

__all__ = [
    "Document",
    "load_gsc_plus",
    "load_biocreative_viii_train",
    "load_biocreative_viii_dev",
    "load_case_reports",
    "load_consult",
]

# Matches GSC+ annotation lines like:  [12::24]\tHP_0000256\tmacrocephaly
_GSC_LINE = re.compile(r"\[(\d+)::(\d+)\]\t(HP_\d+)")


@dataclass(frozen=True)
class Document:
    """One document: an id, its full text, and its gold annotations."""

    doc_id: str
    text: str
    annotations: list[Annotation]

    @property
    def hpo_ids(self) -> set[str]:
        return {a.hpo_id for a in self.annotations}


def load_gsc_plus(
    root: Path, valid_ids: set[str] | None = None
) -> tuple[list[Document], int]:
    """Load GSC+ from an IHP-style tree: ``root/Text/<id>`` + ``root/Annotations/<id>``.

    Each annotation file has lines matching ``[start::end]<TAB>HP_NNNNNNN`` (the
    ``HP_`` form is converted to ``HP:``).

    If ``valid_ids`` is given, annotations whose HPO ID is not in the current
    ontology release are dropped. Returns ``(documents,
    skipped_count)`` so the caller can log how many gold annotations used
    defunct IDs.
    """
    ann_dir = root / "Annotations"
    text_dir = root / "Text"
    documents: list[Document] = []
    skipped = 0
    for ann_path in sorted(p for p in ann_dir.iterdir() if p.is_file()):
        doc_id = ann_path.stem
        text_path = text_dir / ann_path.name
        if not text_path.exists():
            text_path = text_dir / f"{doc_id}.txt"
        text = text_path.read_text(encoding="utf-8")
        annotations: list[Annotation] = []
        for line in ann_path.read_text(encoding="utf-8").splitlines():
            match = _GSC_LINE.search(line)
            if match is None:
                continue
            start, end = int(match.group(1)), int(match.group(2))
            hpo_id = match.group(3).replace("HP_", "HP:")
            if valid_ids is not None and hpo_id not in valid_ids:
                skipped += 1
                continue
            annotations.append(Annotation(start, end, text[start:end], hpo_id))
        documents.append(Document(doc_id=doc_id, text=text, annotations=annotations))
    return documents, skipped


def _longest_span(spans_field: str) -> tuple[int, int] | None:
    """Return the longest ``start-end`` sub-span from a BC8 ``Spans`` cell.

    Discontinuous spans look like ``6-8,20-33``; the canonical (start, end) is
    the widest sub-span.
    """
    best: tuple[int, int] | None = None
    for part in spans_field.split(","):
        part = part.strip()
        if "-" not in part:
            continue
        raw_start, raw_end = part.split("-", 1)
        start, end = int(raw_start), int(raw_end)
        if best is None or (end - start) > (best[1] - best[0]):
            best = (start, end)
    return best


def _load_bc8(path: Path) -> list[Document]:
    """Load one BC8 Track 3 TSV. Rows are grouped by ``StringID``; ``Term == 'NA'``
    rows are skipped (a string with only NA rows becomes a Document with no
    annotations)."""
    grouped: dict[str, Document] = {}
    order: list[str] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            string_id = row["StringID"]
            if string_id not in grouped:
                grouped[string_id] = Document(string_id, row["Text"], [])
                order.append(string_id)
            if row["Term"] == "NA":
                continue
            span = _longest_span(row.get("Spans", ""))
            if span is None:
                continue
            start, end = span
            mention = row["Text"][start:end] if 0 <= start <= end <= len(row["Text"]) else ""
            grouped[string_id].annotations.append(
                Annotation(start, end, mention, row["Term"], negated=row.get("Negated") == "X")
            )
    return [grouped[string_id] for string_id in order]


def load_biocreative_viii_train(root: Path) -> list[Document]:
    return _load_bc8(root / "train.tsv")


def load_biocreative_viii_dev(root: Path) -> list[Document]:
    return _load_bc8(root / "dev.tsv")


def load_case_reports(path: Path) -> list[Document]:
    """Load the case-report (CSC) corpus JSONL: {doc_id, text, hpo_ids}.

    Gold is term-level (no offsets), so annotations carry the HPO id with empty
    spans — only ``hpo_ids`` is used by the document-level metric.
    """
    documents: list[Document] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        annotations = [Annotation(0, 0, "", h) for h in obj["hpo_ids"]]
        documents.append(Document(obj["doc_id"], obj["text"], annotations))
    return documents


def load_consult(path: Path) -> list[Document]:
    """Load the consult eval JSONL (one document per line)."""
    documents: list[Document] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        annotations = [
            Annotation(a["start"], a["end"], a["text"], a["hpo_id"])
            for a in obj["annotations"]
        ]
        documents.append(Document(obj["doc_id"], obj["text"], annotations))
    return documents
