# Common workflows. `make help` lists them.
PY ?= python3
RESULTS ?= results
COHORT ?= cohort_synthetic
PRIMARY ?= mortality_30d

.PHONY: help install test theory cohort analysis all paper clean

help:
	@grep -E '^[a-z-]+:.*?## .*$$' $(MAKEFILE_LIST) \
	  | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

install:  ## editable install with dev extras
	$(PY) -m pip install -e ".[dev]"

test:  ## full test suite (~20 min; includes a MIMIC extraction dry run)
	$(PY) -m pytest -q

theory:  ## numerically certify every theorem
	$(PY) -m infogain.theory.verify --out $(RESULTS)/theory/verification.json

cohort:  ## simulate a MIMIC-shaped cohort with exact ground truth
	$(PY) scripts/make_synthetic_cohort.py --n 50000 --out data/$(COHORT)

analysis:  ## main analysis for the primary endpoint
	$(PY) -m infogain.experiments.run_cohort --cohort data/$(COHORT) \
	  --outcome $(PRIMARY) --out $(RESULTS)/$(COHORT)/$(PRIMARY)

all:  ## everything the paper reports (~4 h on 4 cores)
	$(PY) scripts/run_all.py

paper:  ## regenerate numbers and tables, then build both PDFs
	bash scripts/build_paper.sh $(RESULTS) $(COHORT) $(PRIMARY)

clean:  ## remove build artefacts (keeps data/ and results/)
	rm -rf paper/*.aux paper/*.bbl paper/*.blg paper/*.log paper/*.out \
	       paper/*.fls paper/*.fdb_latexmk paper/*.pdf .pytest_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
