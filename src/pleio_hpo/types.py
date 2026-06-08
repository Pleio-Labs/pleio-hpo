"""Public result types for pleio-hpo.

These are the only data structures, besides ``Annotator`` itself, that the
package promises to keep stable across v1.x releases.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field
from typing import Literal

ScoreSource = Literal["lexical", "embedding"]
# Back-compat alias for the pre-1.0 name; prefer ``ScoreSource``.
ConfidenceSource = ScoreSource
ContextFlag = Literal["negated", "family_history", "uncertain"]


@dataclass(frozen=True)
class Code:
    """A single HPO code returned by the annotator.

    ``score`` is an opaque per-source strength signal, NOT a calibrated
    probability of correctness: lexical hits score a flat 1.0 (exact dictionary
    match), embedding hits carry their cosine similarity, and a code matched by
    both takes the max. Do not interpret it as P(correct) or compare scores
    across sources as if they were on one scale.
    """

    hpo_id: str
    label: str
    spans: list[tuple[int, int]]
    score: float
    score_source: ScoreSource
    context_flags: frozenset[ContextFlag] = field(default_factory=frozenset)

    @property
    def confidence(self) -> float:
        """Deprecated alias for :attr:`score` (removed in 2.0)."""
        warnings.warn(
            "Code.confidence is deprecated; use Code.score "
            "(it was never a calibrated probability).",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.score

    @property
    def confidence_source(self) -> ScoreSource:
        """Deprecated alias for :attr:`score_source` (removed in 2.0)."""
        warnings.warn(
            "Code.confidence_source is deprecated; use Code.score_source.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.score_source

    def to_dict(self) -> dict[str, object]:
        return {
            "hpo_id": self.hpo_id,
            "label": self.label,
            "spans": [list(span) for span in self.spans],
            "score": self.score,
            "score_source": self.score_source,
            "context_flags": sorted(self.context_flags),
        }


@dataclass(frozen=True)
class AnnotationResult:
    """The full result of annotating one piece of text."""

    codes: list[Code]
    text: str
    annotator_version: str
    hpo_version: str

    def to_dict(self) -> dict[str, object]:
        return {
            "annotator_version": self.annotator_version,
            "hpo_version": self.hpo_version,
            "text": self.text,
            "codes": [code.to_dict() for code in self.codes],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)

    def to_tsv(self) -> str:
        header = "hpo_id\tlabel\tscore\tscore_source\tspans\tcontext_flags"
        rows = [header]
        for code in self.codes:
            spans = ",".join(f"{start}:{end}" for start, end in code.spans)
            flags = ",".join(sorted(code.context_flags))
            rows.append(
                f"{code.hpo_id}\t{code.label}\t{code.score:.2f}\t"
                f"{code.score_source}\t{spans}\t{flags}"
            )
        return "\n".join(rows)
