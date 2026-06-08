"""The public Annotator class — orchestrates the full pipeline.

Stages: lexical match -> embedding match -> validator ->
context flags. Lexical hits are high-confidence and skip the validator;
embedding hits go through it. Context flags filter the final set.

"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pleio_hpo.types import AnnotationResult, Code, ContextFlag, ScoreSource

__all__ = ["Annotator"]

logger = logging.getLogger("pleio_hpo")


def _overlaps(a0: int, a1: int, b0: int, b1: int) -> bool:
    return a0 < b1 and a1 > b0


@dataclass
class _Hit:
    hpo_id: str
    start: int
    end: int
    score: float
    source: ScoreSource
    flags: frozenset[ContextFlag] = field(default_factory=frozenset)


class Annotator:
    """Maps free-text clinical notes to HPO codes.

    All heavyweight resources (embedding model, validator, spaCy) are loaded
    lazily on the first call to :meth:`annotate` and cached on the instance.
    """

    def __init__(
        self,
        *,
        hpo_release: str | None = None,
        embedding_model: str | None = None,
        embedding_index_path: str | Path | None = None,
        validator_path: str | Path | None = None,
        use_validator: bool = True,
        use_token_set: bool = True,
        embedding_threshold: float = 0.8,
        validator_threshold: float = 0.5,
        log_level: str | None = None,
    ) -> None:
        self._hpo_release = hpo_release
        self._embedding_model = embedding_model
        self._embedding_index_path = embedding_index_path
        self._validator_path = validator_path
        self._use_validator = use_validator
        self._use_token_set = use_token_set
        self._embedding_threshold = embedding_threshold
        self._validator_threshold = validator_threshold
        # Explicit arg wins; else PLEIO_HPO_LOG_LEVEL; else WARNING.
        logger.setLevel(log_level or os.environ.get("PLEIO_HPO_LOG_LEVEL", "WARNING"))

        self._loaded = False
        self._automaton: Any = None
        self._lemma_index: Any = None
        self._token_set_index: Any = None
        self._pheno_ids: set[str] = set()
        self._meta: dict[str, tuple[str, str]] = {}
        self._embedder: Any = None
        self._index: Any = None
        self._validator: Any = None
        self._hpo_version = "unknown"

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        from pleio_hpo.embed import DEFAULT_MODEL, BiomedEmbedder
        from pleio_hpo.embed_match import default_index_path, load_index
        from pleio_hpo.hpo import (
            bundled_hpo_version,
            default_hpo_path,
            load_terms,
            phenotypic_abnormality_ids,
        )
        from pleio_hpo.match import (
            build_automaton,
            build_lemma_index,
            build_token_set_index,
        )

        logger.info("Loading HPO ontology and lexical automaton...")
        terms = load_terms(default_hpo_path())
        # Candidate vocabulary is restricted to phenotypic abnormalities; modifier /
        # inheritance / history terms ("Bilateral", "Mild", "All") are not annotated.
        self._pheno_ids = phenotypic_abnormality_ids(terms)
        pheno_terms = [t for t in terms if t.id in self._pheno_ids]
        self._automaton = build_automaton(pheno_terms)
        self._lemma_index = build_lemma_index(pheno_terms)
        self._token_set_index = build_token_set_index(pheno_terms)
        self._meta = {t.id: (t.label, t.definition) for t in terms}
        self._hpo_version = bundled_hpo_version()

        self._embedder = BiomedEmbedder(self._embedding_model or DEFAULT_MODEL)
        index_path = (
            Path(self._embedding_index_path)
            if self._embedding_index_path
            else default_index_path()
        )
        if not index_path.exists():
            raise FileNotFoundError(
                f"Embedding index not found at {index_path}. "
                "Fetch it with `pleio-hpo download`."
            )
        self._index = load_index(index_path)

        if self._use_validator:
            from pleio_hpo.validator import Validator, default_validator_path

            path = (
                Path(self._validator_path)
                if self._validator_path
                else default_validator_path()
            )
            if not path.exists():
                raise FileNotFoundError(
                    f"Validator checkpoint not found at {path}. "
                    "Fetch it with `pleio-hpo download`."
                )
            self._validator = Validator(path, threshold=self._validator_threshold)

        self._loaded = True

    def annotate(
        self,
        text: str,
        *,
        include_negated: bool = False,
        include_family_history: bool = False,
        include_uncertain: bool = True,
    ) -> AnnotationResult:
        self._ensure_loaded()
        from pleio_hpo.context import detect
        from pleio_hpo.embed_match import match as embed_match
        from pleio_hpo.match import lemma_match, token_set_match
        from pleio_hpo.match import match as lexical_match
        from pleio_hpo.preprocess import noun_chunks, sentences
        from pleio_hpo.validator import ValidatorCandidate

        sents = sentences(text)

        def _candidate(hpo_id: str, start: int, end: int, surface: str) -> ValidatorCandidate:
            sentence = next((st for ss, se, st in sents if ss <= start and end <= se), text)
            label, definition = self._meta.get(hpo_id, (surface, ""))
            off = sentence.find(surface)
            off = off if off >= 0 else 0
            return ValidatorCandidate(sentence, off, off + len(surface), label, definition)

        # Stage 1 — exact lexical hits (high-confidence, skip validator).
        exact_hits = lexical_match(text, self._automaton)
        hits: list[_Hit] = [
            _Hit(h.hpo_id, h.start, h.end, 1.0, "lexical") for h in exact_hits
        ]
        lex_spans: list[tuple[int, int]] = [(h.start, h.end) for h in exact_hits]

        # Stage 1b — morphology-aware (singularized) matches for plural phrasing
        # ("ear anomalies" -> "ear anomaly"). Less certain than exact matches, so
        # they are gated by the validator when it is enabled.
        seen = {(h.hpo_id, h.start, h.end) for h in exact_hits}
        lemma_only = [
            h for h in lemma_match(text, self._lemma_index)
            if (h.hpo_id, h.start, h.end) not in seen
        ]
        if lemma_only and self._use_validator and self._validator is not None:
            scores = self._validator.score(
                [_candidate(h.hpo_id, h.start, h.end, h.text) for h in lemma_only]
            )
            lemma_only = [h for h, s in zip(lemma_only, scores, strict=True) if s.keep]
        hits.extend(_Hit(h.hpo_id, h.start, h.end, 1.0, "lexical") for h in lemma_only)
        lex_spans.extend((h.start, h.end) for h in lemma_only)

        # Stage 2 — embedding over noun chunks not already covered lexically.
        chunks: list[tuple[int, int, str, str]] = []
        for cs, ce, ctext in noun_chunks(text):
            if any(_overlaps(cs, ce, s, e) for s, e in lex_spans):
                continue
            sentence = next((st for ss, se, st in sents if ss <= cs and ce <= se), text)
            chunks.append((cs, ce, ctext, sentence))
        emb_hits = embed_match(
            chunks,
            self._embedder,
            self._index,
            threshold=self._embedding_threshold,
            top_k=1,
        )
        # Embedding nearest-neighbour can land on a non-phenotype term; drop those
        # to keep the output vocabulary aligned with the lexical automaton.
        emb_hits = [eh for eh in emb_hits if eh.hpo_id in self._pheno_ids]

        # Stage 3 — validator filters embedding hits only.
        if self._use_validator and emb_hits and self._validator is not None:
            cands = []
            for eh in emb_hits:
                label, definition = self._meta.get(eh.hpo_id, (eh.text, ""))
                off = eh.sentence.find(eh.text)
                off = off if off >= 0 else 0
                cands.append(
                    ValidatorCandidate(
                        eh.sentence, off, off + len(eh.text), label, definition
                    )
                )
            scores = self._validator.score(cands)
            emb_hits = [eh for eh, s in zip(emb_hits, scores, strict=True) if s.keep]

        hits.extend(
            _Hit(eh.hpo_id, eh.start, eh.end, eh.score, "embedding") for eh in emb_hits
        )

        # Stage 2b — order-free (token-set) lexical matches for reordered or
        # scattered phrasings the exact/lemma matchers miss ("nervous system
        # tumors" -> "Neoplasm of the nervous system"). Additive over the stages
        # above (it does not consume spans from the embedding stage); only ids not
        # already predicted are considered. Like lemma hits these are less certain
        # than exact matches, so they are gated by the validator when enabled.
        existing_ids = {h.hpo_id for h in hits}
        ts_hits = [
            h for h in token_set_match(text, self._token_set_index)
            if h.hpo_id not in existing_ids and h.hpo_id in self._pheno_ids
        ] if self._use_token_set else []
        if ts_hits and self._use_validator and self._validator is not None:
            scores = self._validator.score(
                [_candidate(h.hpo_id, h.start, h.end, h.text) for h in ts_hits]
            )
            ts_hits = [h for h, s in zip(ts_hits, scores, strict=True) if s.keep]
        hits.extend(_Hit(h.hpo_id, h.start, h.end, 1.0, "lexical") for h in ts_hits)

        # Stage 4 — context flags, per-hit filtering.
        excluded: set[ContextFlag] = set()
        if not include_negated:
            excluded.add("negated")
        if not include_family_history:
            excluded.add("family_history")
        if not include_uncertain:
            excluded.add("uncertain")

        kept: list[_Hit] = []
        for hit in hits:
            flags = detect(text, (hit.start, hit.end))
            present: set[ContextFlag] = set()
            if flags.negated:
                present.add("negated")
            if flags.family_history:
                present.add("family_history")
            if flags.uncertain:
                present.add("uncertain")
            if present & excluded:
                continue
            hit.flags = frozenset(present)
            kept.append(hit)

        codes = self._group_into_codes(kept)
        return AnnotationResult(
            codes=codes,
            text=text,
            annotator_version=self.version,
            hpo_version=self._hpo_version,
        )

    def _group_into_codes(self, hits: list[_Hit]) -> list[Code]:
        by_id: dict[str, list[_Hit]] = {}
        for hit in hits:
            by_id.setdefault(hit.hpo_id, []).append(hit)

        codes: list[Code] = []
        for hpo_id, group in by_id.items():
            label = self._meta.get(hpo_id, (hpo_id, ""))[0]
            spans = [(h.start, h.end) for h in group]
            if any(h.source == "lexical" for h in group):
                score: float = 1.0
                source: ScoreSource = "lexical"
            else:
                score = max(h.score for h in group)
                source = "embedding"
            flags: set[ContextFlag] = set()
            for h in group:
                flags |= h.flags
            codes.append(
                Code(
                    hpo_id=hpo_id,
                    label=label,
                    spans=spans,
                    score=score,
                    score_source=source,
                    context_flags=frozenset(flags),
                )
            )
        codes.sort(key=lambda c: min(s[0] for s in c.spans))
        return codes

    @property
    def version(self) -> str:
        from pleio_hpo import __version__

        return __version__

    @property
    def hpo_version(self) -> str:
        if not self._loaded:
            from pleio_hpo.hpo import bundled_hpo_version

            return bundled_hpo_version()
        return self._hpo_version

    @property
    def model_versions(self) -> dict[str, str]:
        from pleio_hpo.embed import DEFAULT_MODEL

        return {
            "hpo": self.hpo_version,
            "embedding_model": self._embedding_model or DEFAULT_MODEL,
            "validator": "disabled" if not self._use_validator else "v1.0",
        }
