"""04 — Context filtering. Control negated / family-history / uncertain findings.

    python examples/04-context-filtering.py

Each annotate() call decides which context-flagged findings to KEEP. Defaults:
  include_negated=False           (drop "no microcephaly")
  include_family_history=False    (drop "mother had seizures")
  include_uncertain=True          (keep "possible hypotonia")
"""

from __future__ import annotations

from pleio_hpo import Annotator

annotator = Annotator()

text = (
    "No microcephaly. The patient has macrocephaly. "
    "Mother had seizures. Possible hypotonia."
)


def show(label: str, **kwargs: bool) -> None:
    result = annotator.annotate(text, **kwargs)
    codes = ", ".join(
        f"{c.label}"
        + (f" [{', '.join(sorted(c.context_flags))}]" if c.context_flags else "")
        for c in result.codes
    )
    print(f"{label}:\n  {codes or '(none)'}\n")


show("defaults (negated & family-history dropped, uncertain kept)")
show("include negated too", include_negated=True)
show("include family history too", include_family_history=True)
show("drop uncertain", include_uncertain=False)
show("include everything", include_negated=True, include_family_history=True, include_uncertain=True)
