#!/usr/bin/env bash
# Regenerate every number and table from the results tree, then build both PDFs.
# Safe to run against a partial results tree: unavailable quantities typeset as
# a visible [?name] marker and missing figures as a labelled placeholder box.
set -euo pipefail
cd "$(dirname "$0")/.."

RESULTS="${1:-results}"
COHORT="${2:-cohort_synthetic}"
PRIMARY="${3:-mortality_30d}"

python3 scripts/make_paper_numbers.py --results "$RESULTS" --cohort-name "$COHORT" --primary "$PRIMARY"
python3 scripts/make_paper_tables.py  --results "$RESULTS" --cohort-name "$COHORT" --primary "$PRIMARY"

cd paper
latexmk -pdf -interaction=nonstopmode main.tex
latexmk -pdf -interaction=nonstopmode supplementary.tex
echo "built: paper/main.pdf, paper/supplementary.pdf"
