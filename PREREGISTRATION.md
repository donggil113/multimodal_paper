# INFOGAIN — preregistered analysis plan for the MIMIC-IV multimodal cohort

**Status: written before any access to the data.** At the time of writing this
repository has never held a MIMIC-IV record. PhysioNet requires a credentialed
account and a signed data use agreement, and in the environment where this plan
was written outbound access to `physionet.org` is additionally refused at the
network layer. Every hypothesis below is therefore stated in ignorance of its
answer, which is the only condition under which stating it is worth anything.

The point of preregistering is narrow and specific. The estimator in this
repository produces, for each modality and endpoint, a number that can come out
redundant or synergistic. With four modalities and three endpoints there are
enough such numbers that some pattern will look striking after the fact. Fixing
the direction in advance is what separates a finding from a description.

---

## 1. Cohort

**Anchor.** One row per electrocardiogram in MIMIC-IV-ECG, at the time the
electrocardiogram was recorded (`record_list.ecg_time`). Not per admission: the
question is what a clinician holding an ECG should order next, and that decision
is made at the ECG, which may be out-patient.

**Inclusion.** An ECG is eligible when, within ±24 hours of `ecg_time`, the
patient has

- at least one of each required laboratory group (§2), and
- a resolvable age and sex,

and the age at `ecg_time` is ≥ 18.

**Exclusion.** Patients on chronic dialysis are excluded from the AKI endpoint
only (ICD-coded ESRD or a dialysis procedure before the anchor), not from the
cohort.

**Anticipated size.** MIMIC-IV-ECG holds ~800,000 ECGs over ~160,000 patients.
We do not predict the yield after the laboratory requirement and will report it
as found. Nothing below is conditioned on reaching a particular n; if the
overlap (§5) is too small the answer is "not estimable at this overlap", which
this project has already shown is the binding constraint and the honest report.

**Splits.** Grouped by `subject_id` throughout. A patient contributing several
ECGs contributes them all to the same split. This is not optional: ECGs from one
patient are not independent, and a random row split would leak.

## 2. Modalities

| block | contents | source |
|---|---|---|
| `demo` | age, sex, insurance, admission type where present | `patients`, `admissions` |
| `labs` | BNP or NT-proBNP; troponin T or I; creatinine; sodium, potassium, chloride, bicarbonate | `labevents` via `d_labitems` |
| `ecg` | see below | MIMIC-IV-ECG |

Laboratory values are taken as the measurement closest in time to the anchor
within ±24 h, with a per-analyte observed indicator, so "not measured" and
"measured and normal" stay distinguishable.

**ECG representation, stage 1:** the interval and axis measurements in
`machine_measurements` (RR, PR, QRS, QT, QTc by Bazett, P/QRS/T axes), each with
a measured indicator.

**ECG representation, stage 2:** a waveform encoder over the raw 12-lead signal,
either a PTB-XL–pretrained encoder or self-supervised on the MIMIC training
split only. The comparison of stage 1 against stage 2 is itself reported,
because V-information is relative to the predictor family and a richer ECG
representation can only raise the ECG's estimated value.

**Stage 3 (chest radiograph):** MIMIC-CXR studies within ±24 h of the anchor,
embedded by a frozen public chest-radiograph foundation model. Frozen, so that
the image encoder cannot be tuned to the outcome and inflate the estimate.

## 3. Endpoints

| endpoint | definition |
|---|---|
| `mortality_30d` | death within 30 days of the anchor, from `patients.dod` and in-hospital death |
| `hf_admission_365d` | any admission within 365 days carrying a heart-failure diagnosis in the CMS code family |
| `aki_7d` | KDIGO: creatinine rise ≥ 0.3 mg/dL in any rolling 48 h window, or ≥ 1.5× the minimum of the preceding 7 days, within 7 days of the anchor |

All clocks start at the end of the acquisition window, not at the anchor, so a
test result cannot predict an event that preceded it.

## 4. Estimation protocol

Fixed before data, and already implemented and tested in this repository:

1. **Split A fits, split B evaluates.** The predictive family is fitted on the
   training folds and `I_V` is evaluated only on held-out rows, grouped by
   subject. The early-stopping validation split is carved out of the training
   folds, so evaluated rows are absent from fitting *and* from model selection.
   `I_V` is an infimum of expected log-loss over the family; estimating an
   infimum on the rows that selected it understates the conditional entropy and
   therefore overstates the information. The magnitude of that bias is measured
   in `infogain.experiments.run_simulation.train_eval_optimism` and reported.
2. **5-fold cross-fitting**, grouped by subject, with multiple training seeds;
   between-seed variance enters every interval rather than being averaged away.
3. **Bootstrap confidence intervals**, resampled by subject.
4. **Learning-curve bias correction** reported alongside, never instead of, the
   raw estimate.
5. The unique / redundant / synergistic decomposition is tabulated per endpoint.

No hyperparameter, architecture, or mask distribution will be selected using the
evaluation split. The family is frozen to the configuration this repository
ships before the first real-data fit.

## 5. Hypotheses

Stated directionally. Each names what would falsify it.

### H_B1 — attention cross-modal capacity (simulator, not MIMIC)

With no explicit second-order term, recovery of a known synergistic interaction
increases monotonically in the number of attention heads (1, 2, 4, 8, 16) and
saturates. **Falsified by** a flat curve (heads irrelevant), or a curve that
does not saturate by 16 heads, or non-monotonicity beyond noise. Confound to
report: with embedding width fixed at 48, head count trades against per-head
dimension (48/16 = 3), so a decline at high head count is ambiguous between
saturation and per-head starvation, and a width-scaled arm is run to separate
them.

### H_B2 — regime depends on the endpoint

For the ECG–laboratory pair:

- **30-day mortality: redundancy-dominant.** `R > Syn`, with the interaction
  term's interval excluding zero.
- **365-day heart-failure admission: synergy-dominant.** `Syn > R`, likewise.

The reasoning is that short-horizon mortality is driven by acute physiological
derangement that both modalities see, whereas heart-failure trajectory depends
on a conduction/structural picture and a neurohormonal picture that are
complementary.

**Falsified by** either endpoint showing the opposite ordering with an interval
excluding zero. **Inconclusive** (reported as such, not as support) if either
interval includes zero — which, given this project's own finding that pairwise
interactions need events with *both* modalities acquired, is a live possibility
and will be reported with the overlap count that caused it.

### H_B3 — per-patient gain predicts realised benefit

Rank patients by the prospective per-patient gain Δ_i(labs | ECG observed),
computed without the laboratory result and without the outcome. Compare the top
decile against the bottom decile on the *realised* benefit of adding the
laboratory panel, measured as:

- **primary:** difference in net benefit from adding labs, at thresholds in
  [0.02, 0.30], by decision-curve analysis;
- **secondary:** categorical net reclassification improvement at the endpoint's
  prevalence-matched threshold.

**Prediction:** the top decile's realised benefit exceeds the bottom decile's,
with a bootstrap interval on the difference excluding zero.

**Falsified by** an interval covering zero, or the bottom decile benefiting
more. This is the hypothesis most likely to fail: on the simulator, the
prospective score ranks patients well against the *exact* gain (Spearman 0.43
and 0.60 for the modalities carrying per-patient signal) but the realised,
single-draw quantity is dominated by outcome noise, and the decile contrast is
the design chosen to survive that.

## 6. What will be reported regardless of outcome

- The cohort flow diagram, including every exclusion and its count.
- The overlap count (events with both modalities acquired) for every pair,
  since this project's own simulation established it as the binding constraint
  on detecting synergy.
- All three endpoints and all pairs, not only those that came out significant.
- The estimated decomposition with intervals, including the ones covering zero.
- Any deviation from this plan, with its reason, in a section titled as such.

## 7. Deviations

None yet: no data has been seen.
