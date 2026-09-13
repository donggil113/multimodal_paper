# Implementation notes

The manuscript states the method; this file covers the parts a reader who wants
to *change* something needs, and records the choices that were not obvious.

## How a number gets produced

```
Cohort (real or simulated)
   │
   ├─ fit_family()  ──────── K-fold cross-fitting, grouped by subject_id
   │                          one masked network realises f_S for every S
   │                          → out-of-fold P(Y=1 | x_S) for every subset, per seed
   │
   ├─ pvi(S) = log2 f_S(y|x) − log2 f_∅(y)          per patient, in bits
   │
   ├─ decomposition ─── paired differences of pvi vectors
   ├─ shapley ───────── weighted sum of those differences (per patient)
   ├─ patient gain ──── counterfactual resampling + KL, no labels used
   └─ policy sim ────── each patient's prediction read off the subset bought
```

Every arrow is a difference of pointwise quantities, which is why per-patient
scores, cohort estimates and confidence intervals all come out of the same
object.

## Choices that took thought

**Why the null model is the prevalence, not the empty-mask network.**
`H_V(Y)` is an infimum. The constant predictor at the training-fold base rate is
in the family and is the exact minimiser over the constant sub-family; the
network's own all-absent member is also in the family but only approximately
optimal, and any slack in it inflates *every* subset's information by the same
amount. That slack cancels in differences but not in levels, and we report
levels. `CrossFittedFamily.null_gap_bits()` reports the gap so the choice is
auditable — in our runs it is under 0.005 bits.

**Why attention fusion.** Synergy lives in interactions between modalities. A
concatenate-then-MLP head must discover those inside a generic function
approximator; attention gives every modality a direct multiplicative path to
every other. In the simulation study it recovers roughly twice as much of the
known synergy at the same sample size. Because `I_V` is an infimum, the weaker
head is not *wrong*, only looser — set `FamilyConfig(fusion="concat")` to see the
difference.

**Why masks are sampled with spikes on ∅ and the full set.** The empty-set member
anchors every pointwise value, so a badly fitted null biases all subsets
identically; the full set is the model a conventional multimodal paper reports,
and we want ours comparable to it. Bernoulli dropout alone would under-train
both corners.

**Why PVI is clipped at ±8 bits.** A single catastrophically confident wrong
prediction otherwise dominates the mean, and the range enters every
concentration bound linearly. Eight bits is a 256× likelihood ratio, well beyond
anything a calibrated model produces on these endpoints; the clipped fraction is
reported per subset in `diagnostics.csv`.

**Why differences are computed pairwise.** Both terms of a gain share the same
patients and the same null, so differencing per patient before averaging cancels
the patient-level variance that dominates each term separately. At realistic
cohort sizes this tightens the standard error by roughly an order of magnitude.
`test_paired_gain_is_tighter_than_unpaired` pins the behaviour.

**Why bootstrap intervals are the default and Bernstein is reported alongside.**
The empirical-Bernstein interval is a genuine finite-sample certificate, but its
range term is `7R·log(4/α)/(3(n−1))` with `R = 16` bits, which dominates below
n ≈ 10⁴. BCa is the primary interval; Bernstein backs the headline claims.

**Why the learning-curve exponent is free.** The dominant bias here is *learning*
(a finite training set has not reached the family infimum), not the plug-in bias
the classical `1/m` series corrects. Forcing α = 1 badly over-extrapolates the
interaction terms that carry synergy, which are exactly the terms of interest.

**Why donors come only from training folds.** Otherwise a patient can donate a
counterfactual test result to themselves, and the per-patient score leaks.

## Extending

**Add a modality.** Write an extractor returning a `ModalityBlock` (values,
`observed` mask, optional `delta_hours`), register it in `build_cohort`, and add
a `Modality` entry with its cost. Nothing downstream needs changing: subsets,
decomposition, Shapley and the policy simulator all read `cohort.spec`. Beyond
about six orderable modalities, replace `default_subsets` with a targeted list —
the lattice is 2^|M|.

**Add an endpoint.** Add a function to `infogain/data/outcomes.py` returning a
float Series with `NaN` for non-evaluable rows, and list it in
`CohortConfig.endpoints`. Check the index anchor: an endpoint measured from
discharge (readmission) needs `index_anchor="discharge"` and may legitimately use
discharge summaries; one measured from admission must not.

**Swap in a foundation-model encoder.** Compute per-study embeddings with
whatever encoder you trust, save as `.npz` with `study_id` and `embedding`, and
pass them to `extract_cxr(..., embeddings=...)`. This *raises* `I_V` because it
enlarges the family — that is the intended semantics, and it is why the family
must be reported with the number.

**Change the cost model.** `CostWeights` prices money, turnaround minutes and
invasiveness separately; `CostModel.overrides` takes a local fee schedule. The
reduction study is reported as a frontier precisely so a reader with different
prices can read off their own operating point.

## Runtime

On 4 CPU cores, for a 50,000-encounter cohort with 6 modalities:

| stage | time |
|---|---|
| theory verification | ~2 min |
| cohort simulation with exact ground truth | ~12 min |
| `fit_family` (4 folds × 2 seeds × 130 epochs) | ~15 min per endpoint |
| per-patient gains (4 modalities × 24 draws) | ~3 min per endpoint |
| policy simulation and figures | ~2 min per endpoint |

The `labevents` scan on real MIMIC-IV adds ~20 min once, then caches.

## Gotchas

- `Cohort.load` needs `spec.json`; a directory copied without it will fail late.
- The masked family assumes `ModalityBlock.values` is zero on unobserved rows.
  The dataclass enforces this on construction — do not write into `.values`
  afterwards.
- `GroundTruth` caches per-modality log-likelihoods; call `clear_cache()` if you
  mutate `X` in place (nothing in the repository does).
- `monotone_projection` is a Euclidean projection, not a clip: it can move
  values that were not themselves violations. That is the point — it returns the
  closest coherent lattice — but report the raw values too, as `run_cohort` does.
