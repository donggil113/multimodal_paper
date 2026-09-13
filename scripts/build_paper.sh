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
# The supplement must be built first: main.tex reads supplementary.aux through
# xr to resolve "Supplementary Table S8" and friends. Building main alone leaves
# those as ?? -- visibly wrong, which is the point, but not shippable.
latexmk -pdf -interaction=nonstopmode supplementary.tex
latexmk -pdf -interaction=nonstopmode main.tex
# a second supplement pass, so its own forward references settle against the
# numbers main.tex may have shifted
latexmk -pdf -interaction=nonstopmode supplementary.tex

if grep -q "Reference .* undefined" main.log supplementary.log; then
  echo "WARNING: unresolved cross-references remain:" >&2
  grep -h "Reference .* undefined" main.log supplementary.log | sort -u >&2
  exit 1
fi
echo "built: paper/main.pdf, paper/supplementary.pdf"
