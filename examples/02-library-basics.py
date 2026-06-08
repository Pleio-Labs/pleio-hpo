"""02 — Library basics. Use pleio-hpo from Python.

    python examples/02-library-basics.py

Requires the embedding index + validator (run `pleio-hpo download` once).
"""

from __future__ import annotations

from pleio_hpo import Annotator

# The Annotator loads its models lazily on first annotate() call, then reuses
# them. Construct it once and keep it around (it is not cheap to build).
annotator = Annotator()

result = annotator.annotate(
    "The patient has macrocephaly, hypotonia, and global developmental delay."
)

print(f"text:              {result.text}")
print(f"annotator_version: {result.annotator_version}")
print(f"hpo_version:       {result.hpo_version}")
print(f"codes:             {len(result.codes)}\n")

for code in result.codes:
    flags = f"  [{', '.join(sorted(code.context_flags))}]" if code.context_flags else ""
    # `score` is an opaque per-source strength signal (1.0 for an exact lexical
    # match, cosine similarity for an embedding match) — NOT a calibrated
    # probability. `score_source` is "lexical" or "embedding".
    print(f"  {code.hpo_id}  {code.label:<32} {code.score:.2f} ({code.score_source}){flags}")

# Serialization helpers:
print("\n--- as JSON ---")
print(result.to_json())

print("\n--- as TSV ---")
print(result.to_tsv())
