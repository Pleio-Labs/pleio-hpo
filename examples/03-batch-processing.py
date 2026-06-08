"""03 — Batch processing. Annotate many documents from a CSV.

    python examples/03-batch-processing.py

Reads a CSV with an ``id`` and ``text`` column and writes one output row per
(document, HPO code). Reuse a single Annotator across all documents — model
loading is the expensive part, annotate() itself is cheap.
"""

from __future__ import annotations

import csv
import io

from pleio_hpo import Annotator

# Stand-in for your real CSV file. Replace with: open("notes.csv", newline="")
SAMPLE_CSV = """id,text
doc1,Patient has macrocephaly and seizures.
doc2,Short stature with delayed bone age.
doc3,Bilateral hearing loss; no microcephaly.
"""

annotator = Annotator()

reader = csv.DictReader(io.StringIO(SAMPLE_CSV))
out = io.StringIO()
writer = csv.writer(out)
writer.writerow(["doc_id", "hpo_id", "label", "score", "score_source"])

for row in reader:
    result = annotator.annotate(row["text"])
    for code in result.codes:
        writer.writerow([row["id"], code.hpo_id, code.label, f"{code.score:.3f}", code.score_source])

print(out.getvalue())
# In a real pipeline, write to disk instead:
#   with open("annotations.csv", "w", newline="") as fh:
#       fh.write(out.getvalue())
