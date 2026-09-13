# INFOGAIN

**Decomposing the information value of multimodal diagnostic testing at the
level of the individual patient.**

A test's value is not a property of the test. It depends on what is already
known about the patient in front of you. This repository makes that dependence
computable: given a patient's existing data, it estimates what each *additional*
test would tell you, splits that value into the part the existing data already
contained (redundant) and the part that only exists in combination
(synergistic), and converts both into the units clinical decisions are actually
made in.

The central result is an identity, not an analogy. The information a test adds
equals the integral of the net benefit it adds, over every risk threshold a
clinician might hold:

$$\Delta I \;=\; \int_0^1 \Delta \mathrm{NB}(t)\,\frac{\mathrm{d}t}{t}.$$

Everything else follows from it — a certified lower bound on the discrimination
a test can buy, and a certified upper bound on what skipping it can cost.

---

## What is here

| | |
|---|---|
| **Theory** | `infogain/theory/` — Lemmas L1–L3 and Theorems 1–3, each numerically certified |
| **Estimators** | `infogain/vinfo/` — $\mathcal{V}$-usable information, decomposition, Shapley, per-patient gain, finite-sample bounds |
| **Models** | `infogain/encoders/` — masked multimodal family, cross-fitting, raw ECG/CXR encoders |
| **Data** | `infogain/data/` — MIMIC-IV extraction, endpoint definitions, ground-truth simulator |
| **Clinical** | `infogain/clinical/` — decision curves, cost model, ordering policies, reduction study |
| **Paper** | `paper/` — manuscript and supplementary with complete proofs |

## Quick start

```bash
pip install -e .

# 1. Certify every theoretical claim (~2 min)
python -m infogain.theory.verify

# 2. Simulate a MIMIC-shaped cohort with exact known ground truth
python scripts/make_synthetic_cohort.py --n 50000 --out data/cohort_synthetic

# 3. Run the full analysis for one endpoint
python -m infogain.experiments.run_cohort \
    --cohort data/cohort_synthetic --outcome mortality_30d

# ...or reproduce every number and figure in the paper (~1.5 h on 4 cores)
python scripts/run_all.py
```

For real data, see **[`docs/DATA_ACCESS.md`](docs/DATA_ACCESS.md)**: the pipeline
needs credentialed PhysioNet access to MIMIC-IV, MIMIC-IV-ECG, MIMIC-CXR-JPG and
MIMIC-IV-Note. Once you have it, one command swaps the input and nothing else
changes:

```bash
python -m infogain.data.mimic_cohort --root data/raw --out data/cohort_mimic \
    --split temporal
python scripts/run_all.py --mimic-root data/raw
```

> **Status.** The MIMIC-IV extraction is implemented and configured but has not
> been executed here: PhysioNet credentialing is a prerequisite and cannot be
> automated. All numbers currently in the manuscript come from the simulator,
> which reproduces the cohort's schema, modality dimensions, endpoint
> prevalences and informative test-ordering, and for which the true information
> decomposition is available in closed form. Both inputs run through the
> identical analysis code.

## The idea in three steps

**1. Value is conditional.** Fix a free baseline $S_0$ (demographics, vitals) and
a context $S$ of tests already done. For a candidate modality $m$:

```
marginal      I_m        = I_V(S₀ ∪ m) − I_V(S₀)          what it's worth alone
conditional   U(m | S)   = I_V(S  ∪ m) − I_V(S)           what it's worth to you
redundant     R(m | S)   = (I_m − U)₊                     what you'd pay twice for
synergistic   Syn(m | S) = (U − I_m)₊                     what isolation would miss
```

with `U = I_m − R + Syn` exactly and `R · Syn = 0`. Unlike partial information
decomposition, no term depends on a contested axiom: all are differences of
estimable quantities. Shapley averaging over acquisition orders removes the
remaining dependence on *which* tests happened to come first.

**2. Bits are clinical currency.** Theorem 3 (above) is exact — verified
numerically to a relative error of ~10⁻⁵. It also says *which* bits count: only
those earned at thresholds a clinician would act on, which defines a
**clinically restricted usable information** and is typically a minority of the
total. Theorem 1 turns synergy into a certified lower bound on achievable AUROC
gain (via a stratified ranking argument — AUROC is not a proper scoring rule, so
the unconditional version people assume is false). Theorem 2 turns redundancy
into a guarantee: forgoing `ε` nats cannot cost more than `√(ε/2)/(1−t)` in net
benefit, with a substantially tighter threshold-localised version.

**3. The score is per patient and prospective.**

$$\Delta_i(m \mid S) = \mathbb{E}_{x_m \sim q(\cdot \mid x_i^S)}\,
\mathrm{KL}\big(f_{S\cup m}(\cdot \mid x_i^S, x_m)\;\|\;f_S(\cdot \mid x_i^S)\big)$$

It uses neither the test result nor the outcome, so it is computable at the
moment of the ordering decision. The distribution of what the test *would* have
shown comes from a conditional bootstrap over similar patients who did receive
it — no density model, and an explicit overlap diagnostic instead of silent
extrapolation.

## Design decisions worth knowing about

- **Every estimate is out of fold.** Cross-fitting grouped by patient means each
  patient is scored by models that never saw them, which is what makes
  per-patient scores available for *everyone* rather than for a held-out split.
- **The estimate is a lower bound, by construction.** $I_\mathcal{V}$ is an
  infimum over a predictor family, so a weaker model or a smaller cohort
  understates a test's value and never overstates it. The learning bias is
  measured (not assumed away) by refitting at several training-set sizes and
  extrapolating.
- **Seed variance is not sampling variance.** Re-running with a new seed moves
  the answer; that variance is separated out and added to every interval rather
  than averaged away.
- **Nothing after the index time enters the features.** The temporal policy is an
  explicit object recorded in the cohort metadata. Discharge summaries are
  excluded from admission-anchored endpoints — the largest single leakage source
  in multimodal MIMIC work.
- **Non-evaluable is not negative.** Readmission for a patient who died in
  hospital, AKI for a chronic-dialysis patient: coded missing, never zero.
- **Policies are scored honestly.** A policy's prediction for a patient uses only
  the modalities that policy bought, and is compared against random ordering at
  matched budget and against the risk-band heuristic — not only against
  strawmen.

## Reproducibility

```bash
python -m pytest              # unit + integration tests
python -m infogain.theory.verify   # numerical certification of every theorem
python scripts/run_all.py     # full pipeline
python scripts/make_paper_numbers.py && python scripts/make_paper_tables.py
cd paper && latexmk -pdf main.tex && latexmk -pdf supplementary.tex
```

Every number in the manuscript is a LaTeX macro generated from the results tree,
so the text cannot drift from the run that produced it; a quantity whose stage
has not been run typesets as a visible `[?name]` marker rather than a stale
value.

## Layout

```
infogain/
├── theory/       lemmas.py, verify.py            proofs, numerically certified
├── vinfo/        core.py, decomposition.py, estimators.py,
│                 conditional.py, bounds.py        the estimators
├── encoders/     fusion.py, train.py, waveform.py, image.py
├── data/         mimic_io.py, mimic_cohort.py, mimic_modalities.py,
│                 outcomes.py, synthetic.py, schema.py
├── clinical/     net_benefit.py, policy.py, cost.py
├── experiments/  run_cohort.py, run_simulation.py
└── viz/          figures.py, style.py
configs/          cohort, analysis and cost settings
docs/             DATA_ACCESS.md, METHODS.md
paper/            main.tex, supplementary.tex, refs.bib
scripts/          run_all.py and helpers
tests/
```

## Citation

See `paper/main.tex`. Licensed MIT; the data are not — PhysioNet's terms apply
to anything derived from MIMIC-IV, including cached extracts and fitted weights.
