#!/usr/bin/env bash
# 01 — CLI basics. Annotate clinical text with the `pleio-hpo` command.
#
# First-time setup (fetches the embedding index + validator):
#   pleio-hpo download
#
# Run this file:  bash examples/01-cli-basics.sh
set -euo pipefail

echo "## Annotate a string (default human-readable text output):"
pleio-hpo "The patient presented with macrocephaly and hypotonia."

echo
echo "## TSV output (hpo_id, label, score, score_source, spans, context_flags):"
pleio-hpo "Bilateral sensorineural hearing loss and seizures." --format tsv

echo
echo "## JSON output, written to a file:"
pleio-hpo "Global developmental delay was noted." --format json --output /tmp/pleio_demo.json
cat /tmp/pleio_demo.json

echo
echo "## Read from a file instead of an argument, and include negated findings:"
printf 'No microcephaly. Macroglossia present.\n' > /tmp/pleio_note.txt
pleio-hpo --file /tmp/pleio_note.txt --include-negated --format tsv

echo
echo "## Pipe from stdin:"
echo "Short stature and joint hypermobility." | pleio-hpo --format tsv

echo
echo "## Check which local assets are installed:"
pleio-hpo info
