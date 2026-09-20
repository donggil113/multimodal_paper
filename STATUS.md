# INFOGAIN — claim status

Every falsifiable claim in `paper/main.tex` and `paper/supplementary.tex`, with
what establishes it. Written so that a reviewer, or a future session, can tell at
a glance which sentences rest on a proof, which on a computation, and which on
nothing yet.

**How the inventory was built.** Nine agents read disjoint regions of the two
documents and enumerated every falsifiable claim — formal statements, empirical
assertions, and methodological claims of the form "this procedure has this
property". That produced **405 claims**: 155 methodological, 138 empirical, and
112 formal (25 theorem, 40 proposition, 37 lemma, 10 corollary fragments, which
resolve to the 22 distinct environments in Table A). 135 assert a specific
number. The per-claim classification was then done structurally rather than by
model, because whether a proof environment follows a statement, and whether a
named check covers it, are decidable from the files.

**Status vocabulary.**

| status | meaning |
|---|---|
| proved + verified | a proof environment establishes it AND a named numerical check exercises it |
| proved | a proof environment establishes it; no numerical check |
| proved (inline) | the argument is given in running text, one or two steps, without a `proof` environment |
| cited + verified | the mathematical content is a cited standard result, not reproved here; a consequence is checked numerically |
| **unverified** | asserted, with neither a proof nor a check |

---

## A. Formal results

| location | kind | name | status | proof | numerical check |
|---|---|---|---|---|---|
| `main.tex:276` | theorem | Information--net-benefit identity | proved + verified | `supplementary.tex:412` | Thm3 identity: dI = int dNB dt/t |
| `main.tex:317` | theorem | Synergy $\Rightarrow$ achievable discrimination | proved + verified | `supplementary.tex:298` | Thm1 bound <= achievable gain; Thm1 witness identity (bound == lex gain); Thm1 bound informative (>0.005 AUROC) |
| `main.tex:353` | theorem | Redundancy $\Rightarrow$ certified safe omission | proved + verified | `supplementary.tex:458 (global), :487 (local)` | Thm2 global + local bound dominates dNB |
| `supplementary.tex:98` | proposition | Monotonicity | proved | `supplementary.tex:106` | NONE |
| `supplementary.tex:136` | lemma | Neyman--Pearson dominance | proved | `supplementary.tex:142` | NONE |
| `supplementary.tex:163` | lemma | KS sandwich | proved + verified | `supplementary.tex:169` | 2A_hull-1 >= D+; 2A-1 <= 2 D+ |
| `supplementary.tex:192` | lemma | TV/dTV | proved | `supplementary.tex:197` | NONE |
| `supplementary.tex:220` | lemma | L3 | proved + verified | `supplementary.tex:228` | I <= H(pi) * TV; I >= 8 pi^2 (1-pi)^2 TV^2 (proved); I >= 2 pi (1-pi) TV^2 (sharper, numerical only) |
| `supplementary.tex:272` | corollary | Information bracket for achievable AUROC | proved | `supplementary.tex:278` | NONE |
| `supplementary.tex:286` | theorem | Stratified achievable-AUROC gain | proved + verified | `supplementary.tex:298` | Thm1 bound <= achievable gain; Thm1 witness identity (bound == lex gain) |
| `supplementary.tex:322` | corollary | Information form | proved (inline) | `supplementary.tex:324 (one-step, no proof env)` | NONE |
| `supplementary.tex:366` | lemma | Net benefit of the Bayes rule | proved | `supplementary.tex:373` | NONE |
| `supplementary.tex:380` | lemma | Schervish representation for the binary log-loss | proved | `supplementary.tex:389` | NONE |
| `supplementary.tex:402` | theorem | Information--net-benefit identity | proved + verified | `supplementary.tex:412` | Thm3 identity: dI = int dNB dt/t |
| `supplementary.tex:452` | theorem | Global bound | proved | `supplementary.tex:458` | NONE |
| `supplementary.tex:477` | theorem | Threshold-localised bound | proved + verified | `supplementary.tex:487` | Thm2 local bound dominates dNB |
| `supplementary.tex:503` | corollary | Cohort-level omission | proved (inline, gap) | `supplementary.tex:504 (Jensen, one step)` | NONE |
| `supplementary.tex:511` | corollary | Study planning | **unverified** | `NONE` | NONE |
| `supplementary.tex:525` | proposition | Exactness and disjointness | proved | `supplementary.tex:532` | NONE |
| `supplementary.tex:547` | proposition | Shapley attribution | cited + verified | `Shapley 1953 (uniqueness cited, not reproved)` | shapley_efficiency_residual <= 1.4e-17 across all 4 endpoints |
| `supplementary.tex:569` | theorem | Two-sided bracket | proved + verified | `supplementary.tex:585` | PVI recovers I(X;Y) (truth=0.17312, est=0.17146) |
| `supplementary.tex:631` | proposition | Identification | proved (inline) | `supplementary.tex:639 (standard MAR argument)` | overlap condition (ii) checked: overlap_diagnostics.csv |
**Totals: 9 proved + verified, 8 proved, 3 proved inline, 1 cited + verified, 1 unverified.**

All twelve numerical checks in `results/theory/verification.json` report **zero
violations**; the counts and worst slacks are in Supplementary Table S1.

### The four that are weaker than the table's headline

- **`Study planning` (S:511) — unverified.** It asserts that inverting
  $v^\star$ gives a minimum detectable information gain. Inversion needs
  $v^\star$ strictly monotone in $\varepsilon$, which is neither proved nor
  checked. It is a remark presented as a corollary.
- **`Cohort-level omission` (S:503) — proved inline, with a gap.** The Jensen
  step is correct *given* that the per-patient bound is concave in
  $\varepsilon$. That concavity is asserted in the same sentence, not
  established.
- **`Identification` (S:631) — proved inline, with an untestable premise.** The
  MAR + overlap argument is standard and correctly stated. Overlap (ii) is
  checked per modality (`overlap_diagnostics.csv`, Table S6). **MAR (i) is not
  testable from the data at all**, which the manuscript says, and the MNAR
  sensitivity analysis is the response rather than a resolution.
- **`Shapley attribution` (S:547) — cited, not reproved.** Uniqueness under the
  four axioms is Shapley (1953); the paper does not reprove it and should not.
  What is verified here is efficiency, numerically: the residual is at most
  $1.4\times10^{-17}$ bits across all four endpoints.

---

## B. Experimental claims

The 138 empirical claims are not listed individually — each is a sentence about
one of eight generated artifacts, and the artifact is the unit that can be
checked. Every quoted number in both documents is a macro generated by
`scripts/make_paper_numbers.py` from the results tree; a macro that cannot be
resolved typesets as a visible `\todonum` marker rather than silently
disappearing.

| Results subsection | backing artifact | status |
|---|---|---|
| estimator recovery, learning-curve correction | `results/simulation/tables/recovery.csv` | numerically verified against closed-form truth |
| regime calls by cohort size | `regime_recovery.csv`, `regime_by_size.csv`, `regime_first_correct.csv` | numerically verified |
| overlap governs synergy detection | `synergy_power.csv`, `synergy_power_tree.csv` | numerically verified, 2 cohorts |
| fusion-architecture ablation | `architecture_ablation.csv` | numerically verified, 2 cohorts |
| per-patient targeting vs exact truth | `patient_targeting.csv` | numerically verified |
| decomposition per endpoint | `cohort_synthetic/*/tables/decomposition.csv` | numerically verified, permutation p-values |
| policy reduction at matched AUROC | `policy_frontier.csv`, `policy_comparators.csv` | numerically verified |
| MNAR sensitivity | `results/sensitivity_mnar/` | numerically verified |

**Default architecture.** All numbers were regenerated after the fusion head
changed from attention to concatenation (T1), at 5 training seeds, on identical
data. `results/diff_default_switch.json` records the comparison over 691 numeric
keys: median absolute relative change 2.2%, 45% of values rose, **zero material
sign flips** — all seven sign changes are quantities oscillating about zero.
Four categorical regime calls moved (Supplementary Table S12), three of them
modality pairs that concatenation resolves as synergistic where attention could
not separate them from zero. Two paper claims were rewritten because the new
numbers falsified them: the synergistic-regime recovery sentence, now generated
from `regime_by_size.csv` rather than described in prose, and the per-patient
rank correlations, which fell from 0.43–0.60 to 0.25–0.50.

**The load-bearing caveat.** These are properties of the calibrated simulator,
not of patients. MIMIC-IV requires credentialed PhysioNet access, and in this
environment `physionet.org:443` is additionally refused at the network layer.
Which modality is redundant for which endpoint, and how much testing can be
avoided, are therefore **unverified as clinical claims** — they are verified only
as statements about a generative model we wrote. `PREREGISTRATION.md` fixes the
hypotheses and the analysis plan before that data is seen.

---

## C. Methodological claims

The 155 methodological claims assert that a procedure has a property. The ones
with test coverage:

| claim | established by |
|---|---|
| masking closure makes $I_\mathcal{V}$ monotone on the subset lattice | Prop. S:98 (proved) + monotonicity violations reported per run |
| cross-fitting removes the infimum's optimistic bias | `train_eval_optimism` — measured, not asserted |
| the paired gain estimator is tighter by $1/\sqrt{1-\rho}$ | closed form + `test_paired_gain_*` at $\rho=0.72, 0.97$ |
| the permutation test is calibrated under the null | `test_permutation_pvalue_calibrated_under_the_null` |
| seed variance widens intervals | `test_seed_variance_widens_the_interval` |
| donors come only from training folds | `test_patient_gain_is_nonnegative_and_prospective` |
| the temporal policy admits candidate tests without leaking | `tests/test_mimic_pipeline.py` (17 tests on MIMIC-shaped fixtures) |

Claims **without** a test, flagged as unverified methodology:

- that the decomposition terms are *unobservable* in real data (S:374). The
  paper's own two-sided bracket (S:569) gives computable one-sided guarantees on
  every subset value, and the tower-property consistency check validates the
  sampler without any ground truth. The sentence is too strong as written: what
  is unidentified is the *truth needed to score recovery*, not the terms.
- that the interpretable modality representations cost only "some" usable
  information. The raw-signal path is implemented but has never been run.

---

## D. What this document does not cover

The 405-claim inventory was produced by agents reading the documents; I verified
the 22 formal results and the artifact mapping directly, but did not
individually re-derive each of the 138 empirical sentences. The per-claim
adversarial pass was designed and launched but exhausted the session's agent
budget after 25 of 429 agents, so its verdicts are not in this table. Treat
Table A as verified and Tables B and C as a structural map rather than a
line-by-line audit.

### Citations pending verification

The Related Work section was added from a literature sweep run in this
environment. Most citations resolve against entries that were already in
`refs.bib` or against works I can confirm, but these carry an `UNCERTAIN` marker
in `paper/refs.bib` and must be checked against a primary source before
submission:

| key | what is unconfirmed |
|---|---|
| `yang2025lsmi` | **the work itself** — ICML 2025, not verifiable from here |
| `choi2025icym2i` | **the work itself** — arXiv 2505.16953, not verifiable from here |
| `liang2023pid` | author given names, proceedings details |
| `hewitt2021conditional` | page range |
| `covert2023dfs` | PMLR 202 page range |
| `sadatsafavi2022voi` | volume/issue/pages |
| `basu2007evic` | pages |
| `erion2022coai` | author list is truncated with `others` |

A wrong citation is worse than a missing one, so none of these should survive
into a submission unchecked.

---

Regenerated by hand; not yet automated. Re-run the inventory with
`Workflow({scriptPath: ...infogain-audit-position-mechanism...})` if the
documents change materially.
