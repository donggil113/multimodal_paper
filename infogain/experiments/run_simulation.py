r"""Simulation study: does the estimator recover information it cannot see?

The decomposition claims to separate unique, redundant and synergistic usable
information.  On real data none of those is observable, so the claim is only
testable against a generative model where all three are known exactly.
:mod:`infogain.data.synthetic` provides one -- binary latent factors, Gaussian
read-outs, and therefore closed-form posteriors for every modality subset.

Three questions are answered here, and the answers are the paper's validation
section:

1. **Is the estimator consistent?**  Estimated information should approach the
   truth as the cohort grows.  It approaches from *below*, because
   :math:`I_{\mathcal V}` is an infimum over what a finite training set can
   learn -- so the interesting quantity is the recovery *curve*, not a single
   number, and the learning-curve extrapolation is judged on whether it closes
   the remaining gap.
2. **Does it get the qualitative call right?**  Long before the bits are
   numerically accurate, the estimator has to say *redundant* where the truth is
   redundant and *synergistic* where it is synergistic.  A test that is
   worthless once its neighbour is in hand must be identified as such at
   realistic sample sizes or the method is not deployable.
3. **Do the theorems bind?**  Theorem 1's certified AUROC gain must not exceed
   the achievable gain, and Theorem 2's certified net-benefit loss must not fall
   below the realised loss -- checked here against exact posteriors rather than
   fitted ones.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from infogain.clinical.net_benefit import (
    bayes_value, check_identity, information_gain_nats, safe_omission_bound_local,
)
from infogain.data.synthetic import default_design, generate, ground_truth_table
from infogain.encoders.train import TrainConfig, fit_family
from infogain.theory.lemmas import auroc, auroc_hull, best_stratified_bound
from infogain.utils.io import save_json
from infogain.utils.logging import get_logger
from infogain.vinfo.core import subset_key, v_info_from_pvi
from infogain.vinfo.decomposition import decomposition_table
from infogain.vinfo.estimators import extrapolate_infinite_data
from infogain.viz import figures as F
from infogain.viz.style import save

log = get_logger("infogain.experiments.run_simulation")
LN2 = float(np.log(2.0))


def _probe_subsets(cohort) -> list[frozenset[str]]:
    base = cohort.spec.baseline
    orderable = [m for m in cohort.spec.names if m not in base]
    out = [base, frozenset(cohort.spec.names)]
    out += [base | {m} for m in orderable]
    out += [frozenset(cohort.spec.names) - {m} for m in orderable]
    if len(orderable) >= 2:
        out.append(base | {orderable[0], orderable[1]})
    seen, uniq = set(), []
    for s in out:
        if subset_key(s) not in seen:
            seen.add(subset_key(s))
            uniq.append(s)
    return uniq


def recovery_study(sizes=(4000, 12000, 36000), outcome: str = "mortality_30d",
                   seed: int = 0, train: TrainConfig | None = None) -> pd.DataFrame:
    """Estimated vs exact information at several cohort sizes."""
    train = train or TrainConfig(epochs=110, n_folds=4, seeds=(0, 1), patience=18)
    rows = []
    for n in sizes:
        cohort, gt = generate(n=n, seed=seed)
        t0 = time.time()
        cf = fit_family(cohort, outcome, train)
        log.info("n=%d fitted in %.0fs (null gap %+.5f bits)", n, time.time() - t0,
                 cf.null_gap_bits())
        for s in _probe_subsets(cohort):
            vi = v_info_from_pvi(cf.pvi(s), seed_values=cf.seed_means(s))
            truth = gt.information(outcome, s, respect_observation=True)
            rows.append({"n": n, "subset": subset_key(s), "size": len(s),
                         "est_bits": vi.value, "ci_lo": vi.ci_lo, "ci_hi": vi.ci_hi,
                         "se": vi.se, "truth_bits": truth,
                         "recovered": vi.value / max(truth, 1e-9)})
    return pd.DataFrame(rows)


def add_extrapolation(df: pd.DataFrame) -> pd.DataFrame:
    """Learning-curve-correct each subset and record the corrected estimate."""
    df = df.copy()
    df["corrected_bits"] = np.nan
    df["extrap_alpha"] = np.nan
    for subset, g in df.groupby("subset"):
        g = g.sort_values("n")
        if len(g) < 3:
            continue
        ex = extrapolate_infinite_data(g["n"].to_numpy(float), g["est_bits"].to_numpy(float),
                                       weights=np.clip(g["se"].to_numpy(float), 1e-5, None))
        df.loc[df["subset"] == subset, "corrected_bits"] = ex.i_infinity
        df.loc[df["subset"] == subset, "extrap_alpha"] = ex.alpha
    return df


def regime_of(marginal: float, conditional: float, tol: float = 0.002) -> str:
    """Label a modality redundant, synergistic or independent given a tolerance."""
    if conditional - marginal > tol:
        return "synergistic"
    if marginal - conditional > tol:
        return "redundant"
    return "independent"


def qualitative_study(sizes=(6000, 18000, 50000), outcome: str = "mortality_30d",
                      seed: int = 0, train: TrainConfig | None = None,
                      tol: float = 0.002) -> pd.DataFrame:
    r"""Does the estimated decomposition recover the known regime, and at what cost?

    Reported as a function of cohort size rather than at one size, because the
    answer is not a constant and the variation is the useful part.  Getting the
    bits right is hard; getting the *call* right -- redundant, synergistic,
    independent -- is what a hospital deciding whether to order a test actually
    needs, and it happens at a smaller sample size.  How much smaller, and how
    that depends on the size of the effect, is a planning number for anyone
    designing such a study, so we measure it instead of asserting it.

    Synergistic terms are the last to be recovered: a marginal effect is a main
    effect, while synergy is an interaction between latent factors buried in two
    different noisy read-outs, and interactions need more events.
    """
    train = train or TrainConfig(epochs=110, n_folds=4, seeds=(0, 1), patience=18)
    rows = []
    for n in sizes:
        cohort, gt = generate(n=n, seed=seed)
        t0 = time.time()
        cf = fit_family(cohort, outcome, train)
        log.info("regime study n=%d fitted in %.0fs", n, time.time() - t0)
        est = decomposition_table(cf).set_index("modality")
        truth = ground_truth_table(gt, outcome, baseline=sorted(cohort.spec.baseline),
                                   respect_observation=True).set_index("modality")
        for m in est.index:
            if m not in truth.index:
                continue
            t, e = truth.loc[m], est.loc[m]
            rows.append({
                "n": n, "modality": m,
                "true_marginal": t["marginal"], "est_marginal": e["marginal_bits"],
                "true_conditional": t["conditional"],
                "est_conditional": e["conditional_bits"],
                "true_redundant": t["redundant"], "est_redundant": e["redundant_bits"],
                "true_synergistic": t["synergistic"],
                "est_synergistic": e["synergistic_bits"],
                "true_regime": regime_of(t["marginal"], t["conditional"], tol),
                "est_regime": regime_of(e["marginal_bits"], e["conditional_bits"], tol),
            })
    out = pd.DataFrame(rows)
    out["regime_correct"] = out["true_regime"] == out["est_regime"]
    return out


def regime_accuracy_by_size(qual: pd.DataFrame) -> pd.DataFrame:
    """Accuracy per cohort size, overall and split by the true regime."""
    overall = qual.groupby("n")["regime_correct"].mean().rename("all")
    by_regime = (qual.pivot_table(index="n", columns="true_regime",
                                  values="regime_correct", aggfunc="mean"))
    return pd.concat([overall, by_regime], axis=1).reset_index()


def first_correct_size(qual: pd.DataFrame) -> pd.DataFrame:
    """Smallest cohort size at which each modality's regime is called correctly."""
    rows = []
    for m, g in qual.groupby("modality"):
        g = g.sort_values("n")
        hit = g[g["regime_correct"]]
        rows.append({"modality": m,
                     "true_regime": g["true_regime"].iloc[-1],
                     "true_effect_bits": float(abs(g["true_conditional"].iloc[-1]
                                                   - g["true_marginal"].iloc[-1])),
                     "first_correct_n": int(hit["n"].iloc[0]) if len(hit) else -1})
    return pd.DataFrame(rows).sort_values("true_effect_bits", ascending=False)


def synergy_power(sizes=(18000, 50000), completeness=(False, True),
                  outcome: str = "mortality_30d", seed: int = 0,
                  train: TrainConfig | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    r"""What governs whether a synergistic term is recoverable at all?

    Synergy is the last part of the decomposition to become estimable, and the
    binding constraint is not the cohort size. An interaction between two
    modalities can only be learned from patients who received *both*, so with
    realistic acquisition rates -- 67\% for the electrocardiogram and 57\% for the
    radiograph in our design -- the effective sample is about 40\% of the cohort,
    and the effective *event* count correspondingly smaller. That is the number
    to power a multimodal study on, and it is not the one usually reported.

    The grid crosses cohort size with an all-modalities-acquired variant, which
    separates the two explanations: if size were the constraint, the complete-data
    arm at fixed :math:`n` would look the same as the incomplete one.

    A gradient-boosted-tree arm is included as a reference. It is not a member of
    :math:`\mathcal V` and its numbers are not :math:`I_{\mathcal V}` estimates;
    it is there to show the information is extractable by *something*, so that a
    low estimate reads as "hard to reach" rather than "not present".
    """
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import GroupKFold

    from infogain.vinfo.core import label_logprob, pointwise_v_information

    train = train or TrainConfig(epochs=130, n_folds=4, seeds=(0, 1), patience=18)
    rows, tree_rows = [], []
    for n in sizes:
        for complete in completeness:
            design = default_design()
            if complete:
                design.missingness = {m: (12.0, 0.0) for m in design.missingness}
            cohort, gt = generate(design, n=n, seed=seed)
            base, full = cohort.spec.baseline, frozenset(cohort.spec.names)
            truth = ground_truth_table(gt, outcome, baseline=sorted(base),
                                       respect_observation=True).set_index("modality")
            i_true = gt.information(outcome, set(full), respect_observation=True)
            y = cohort.y(outcome)
            both = (cohort.blocks["ecg"].observed & cohort.blocks["cxr"].observed)
            shared = {"n": n, "complete": complete,
                      "both_present_frac": float(both.mean()),
                      "events_with_both": int((y * both).sum()),
                      "i_full_true": i_true}

            t0 = time.time()
            cf = fit_family(cohort, outcome, train)
            log.info("power study n=%d complete=%s fitted in %.0fs", n, complete,
                     time.time() - t0)
            est = decomposition_table(cf, n_perm=1).set_index("modality")
            i_full = float(cf.pvi(full).mean())
            for m in est.index:
                if m not in truth.index:
                    continue
                t, e = truth.loc[m], est.loc[m]
                rows.append({**shared, "family": "neural", "modality": m,
                             "i_full_est": i_full,
                             "true_marginal": t["marginal"],
                             "est_marginal": e["marginal_bits"],
                             "true_conditional": t["conditional"],
                             "est_conditional": e["conditional_bits"],
                             "true_regime": regime_of(t["marginal"], t["conditional"]),
                             "est_regime": regime_of(e["marginal_bits"],
                                                     e["conditional_bits"])})

            # tree reference on the same folds and the same PVI definition
            names = cohort.spec.names
            cols, span, at = [], {}, 0
            for m in names:
                v = cohort.blocks[m].values.astype(np.float64).copy()
                v[~cohort.blocks[m].observed] = np.nan
                cols.append(v)
                span[m] = (at, at + v.shape[1])
                at += v.shape[1]
            X = np.hstack(cols)
            groups = cohort.index["subject_id"].to_numpy()
            want = [base, full] + [full - {m} for m in cohort.spec.orderable] \
                   + [base | {m} for m in cohort.spec.orderable]
            probs = {subset_key(s): np.zeros(len(y)) for s in want}
            p_null = np.zeros(len(y))
            for tr, te in GroupKFold(n_splits=train.n_folds).split(X, y, groups):
                p_null[te] = y[tr].mean()
                for s in want:
                    idx = np.concatenate([np.arange(*span[m]) for m in sorted(s)])
                    mdl = HistGradientBoostingClassifier(
                        max_iter=400, learning_rate=0.06, l2_regularization=1.0,
                        early_stopping=True, n_iter_no_change=25, random_state=seed)
                    mdl.fit(X[tr][:, idx], y[tr])
                    probs[subset_key(s)][te] = mdl.predict_proba(X[te][:, idx])[:, 1]
            tpvi = lambda ss: pointwise_v_information(  # noqa: E731
                label_logprob(probs[subset_key(ss)], y), label_logprob(p_null, y))
            for m in cohort.spec.orderable:
                t = truth.loc[m]
                tree_rows.append({**shared, "family": "tree", "modality": m,
                                  "i_full_est": float(tpvi(full).mean()),
                                  "true_marginal": t["marginal"],
                                  "est_marginal": float((tpvi(base | {m}) - tpvi(base)).mean()),
                                  "true_conditional": t["conditional"],
                                  "est_conditional": float(
                                      (tpvi(full) - tpvi(full - {m})).mean())})
    out = pd.DataFrame(rows)
    out["regime_correct"] = out["true_regime"] == out["est_regime"]
    return out, pd.DataFrame(tree_rows)


def theorem_study_exact(n: int = 60000, outcome: str = "mortality_30d",
                        seed: int = 0) -> dict:
    """Check Theorems 1-3 against *exact* posteriors from the simulator."""
    cohort, gt = generate(n=n, seed=seed)
    y = gt.y[outcome]
    base = sorted(cohort.spec.baseline)
    full = set(cohort.spec.names)
    p_full = gt.posterior(outcome, full, respect_observation=False)

    thm1, thm2 = [], []
    for m in (set(cohort.spec.names) - set(base)):
        ctx = full - {m}
        p_ctx = gt.posterior(outcome, ctx, respect_observation=False)
        b = best_stratified_bound(y, p_ctx, p_full)
        # Compare the bound against the achievable gain measured on the *same*
        # rows. With the honest split the bound lives on the evaluation half, and
        # comparing it to a full-cohort gain reports half-sample noise as a
        # theorem violation -- which it did, for one modality, by 0.001 AUROC.
        thm1.append({"modality": m, "bound": b.bound,
                     "achievable_gain": b.achievable_gain,
                     "lexicographic_gain": b.lexicographic_gain,
                     "identity_residual": b.identity_residual,
                     "observed_gain": b.observed_gain, "n_eval": b.n_eval,
                     "holds": bool(b.bound <= b.achievable_gain + 1e-9)})
        forgone = information_gain_nats(p_full, p_ctx) / LN2
        for t in (0.02, 0.05, 0.10, 0.20):
            tt = np.array([t])
            realized = float((bayes_value(p_full, tt) - bayes_value(p_ctx, tt))[0] / (1 - t))
            cert = safe_omission_bound_local(forgone, t)
            thm2.append({"modality": m, "threshold": t, "forgone_bits": forgone,
                         "realized_nb_loss": realized, "certified_bound": cert,
                         "holds": bool(cert >= realized - 1e-9)})

    p_base = gt.posterior(outcome, set(base), respect_observation=False)
    from infogain.clinical.net_benefit import nest_calibrate

    ident = check_identity(p_full, nest_calibrate(p_full, p_base))
    return {"theorem1": thm1, "theorem2": thm2,
            "theorem3": {"kl_bits": ident.kl_nats / LN2,
                         "integral_bits": ident.integral_nats / LN2,
                         "rel_error": ident.rel_error, "passes": bool(ident.passes)},
            "n": n, "outcome": outcome}


#: Subsets whose true information is below this carry no usable signal, and a
#: "percent recovered" computed against them is a ratio of two noise terms --
#: which is how the raw table shows 180% recovery on a subset worth 0.008 bits.
#: Recovery statistics are reported over subsets above the floor, with the count
#: excluded stated alongside.
RECOVERY_FLOOR_BITS = 0.01


def summarize(rec: pd.DataFrame, qual: pd.DataFrame, thm: dict,
              sizes: tuple[int, ...], outcome: str,
              floor: float = RECOVERY_FLOOR_BITS,
              abl: pd.DataFrame | None = None,
              ref: pd.DataFrame | None = None) -> dict:
    """Assemble the simulation summary, guarding the near-zero-truth subsets."""
    big = rec[rec["n"] == max(sizes)]
    ok = big[big["truth_bits"] >= floor]
    corrected = (ok["corrected_bits"] / ok["truth_bits"].clip(lower=1e-9)
                 if len(ok) else pd.Series(dtype=float))
    return {
        "sizes": list(sizes), "outcome": outcome,
        "recovery_floor_bits": floor,
        "n_subsets_scored": int(len(ok)),
        "n_subsets_below_floor": int(len(big) - len(ok)),
        "recovery_at_largest_n": float(ok["recovered"].median()) if len(ok) else float("nan"),
        "recovery_after_correction": float(corrected.median()) if len(ok) else float("nan"),
        "recovery_full_panel": float(
            big.loc[big["size"].idxmax(), "recovered"]) if len(big) else float("nan"),
        "mean_abs_error_bits": float((big["est_bits"] - big["truth_bits"]).abs().mean()),
        "regime_accuracy": float(
            qual[qual["n"] == qual["n"].max()]["regime_correct"].mean()),
        **({} if abl is None else _power_summary(abl, ref)),
        "regime_accuracy_by_size": regime_accuracy_by_size(qual).to_dict("records"),
        "regime_first_correct": first_correct_size(qual).to_dict("records"),
        "regime_max_n": int(qual["n"].max()),
        "regime_table": qual.to_dict("records"),
        "theorem1_all_hold": bool(all(r["holds"] for r in thm["theorem1"])),
        "theorem2_all_hold": bool(all(r["holds"] for r in thm["theorem2"])),
        "theorem3": thm["theorem3"],
    }


def _power_summary(abl: pd.DataFrame, ref: pd.DataFrame | None) -> dict:
    """Condense the synergy power grid into the quantities the paper quotes."""
    syn = abl[abl["true_regime"] == "synergistic"].copy()
    syn["recovered"] = syn["est_conditional"] / syn["true_conditional"].clip(lower=1e-9)
    by = (syn.groupby(["n", "complete"])
             .agg(events_with_both=("events_with_both", "first"),
                  both_present_frac=("both_present_frac", "first"),
                  syn_recovered=("recovered", "mean"),
                  i_full_recovered=("i_full_est", "first"),
                  i_full_true=("i_full_true", "first"))
             .reset_index())
    by["i_full_recovered"] = by["i_full_recovered"] / by["i_full_true"].clip(lower=1e-9)
    out = {"power_grid": by.to_dict("records"),
           "power_regime_accuracy": float(abl["regime_correct"].mean())}
    lo = by.loc[by["events_with_both"].idxmin()]
    hi = by.loc[by["events_with_both"].idxmax()]
    out.update({
        "power_events_min": int(lo["events_with_both"]),
        "power_events_max": int(hi["events_with_both"]),
        "power_syn_recovered_min": float(lo["syn_recovered"]),
        "power_syn_recovered_max": float(hi["syn_recovered"]),
        "power_both_present_frac": float(
            by.loc[~by["complete"], "both_present_frac"].mean()),
    })
    if ref is not None and len(ref):
        r = ref.copy()
        r["recovered"] = r["est_conditional"] / r["true_conditional"].clip(lower=1e-9)
        rs = r[r["true_conditional"] > 0.005]
        out["tree_syn_recovered_max"] = float(rs["recovered"].max()) if len(rs) else float("nan")
    return out


def rebuild_summary(out: Path, outcome: str) -> dict:
    """Recompute summary.json from the saved tables, without refitting anything."""
    rec = pd.read_csv(out / "tables" / "recovery.csv")
    qual = pd.read_csv(out / "tables" / "regime_recovery.csv")
    t1 = pd.read_csv(out / "tables" / "exact_theorem1.csv")
    t2 = pd.read_csv(out / "tables" / "exact_theorem2.csv")
    thm = {"theorem1": t1.to_dict("records"), "theorem2": t2.to_dict("records"),
           "theorem3": json.loads((out / "summary.json").read_text()).get("theorem3", {})
           if (out / "summary.json").exists() else {}}
    sizes = tuple(sorted(rec["n"].unique()))
    abl_p = out / "tables" / "synergy_power.csv"
    ref_p = out / "tables" / "synergy_power_tree.csv"
    abl = pd.read_csv(abl_p) if abl_p.exists() else None
    ref = pd.read_csv(ref_p) if ref_p.exists() else None
    summary = summarize(rec, qual, thm, sizes, outcome, abl=abl, ref=ref)
    save_json(summary, out / "summary.json")
    return summary


def patient_targeting_study(n: int = 50000, outcome: str = "mortality_30d",
                            epochs: int = 130, folds: int = 4,
                            seeds: tuple[int, ...] = (0, 1),
                            seed: int = 0) -> pd.DataFrame:
    r"""Does the prospective score rank *patients* correctly, not just average?

    The cohort-level consistency check (:func:`infogain.vinfo.conditional.
    consistency_check`) compares the mean of the prospective score against the
    mean realised PVI gain, and they agree.  Their *rank* correlation is close to
    zero, which looks alarming until one asks what the retrospective quantity is:
    :math:`\mathrm{pvi}_i(S\cup m) - \mathrm{pvi}_i(S)` is a single-draw
    log-likelihood ratio at the one outcome this patient happened to have.  Its
    conditional expectation given :math:`x_i` is the gain, which is why the means
    match, but for a rare endpoint the draw dominates the signal, so it is a poor
    per-patient yardstick no matter how good the score is.

    This function measures that directly.  ``spearman_truth_vs_retro`` is the
    ceiling: the rank correlation the *exact* per-patient gain achieves against
    the retrospective quantity.  If the estimator's correlation with the
    retrospective gain is near that ceiling, the near-zero number in the
    consistency table is a property of the yardstick and not a defect.
    ``spearman_est_vs_truth`` is the question that actually matters for a policy
    that orders tests, and ``capture_at_10`` is its decision-relevant form: of
    all the information an oracle could buy with a budget of 10% of patients, how
    much does ordering by the score actually buy?
    """
    from scipy.stats import spearmanr

    from infogain.vinfo.conditional import (GainConfig, patient_expected_gain,
                                            retrospective_gain)

    cohort, gt = generate(n=n, seed=seed)
    base = sorted(cohort.spec.baseline)
    cf = fit_family(cohort, outcome,
                    TrainConfig(epochs=epochs, n_folds=folds, seeds=tuple(seeds),
                                patience=18), keep_models=True)
    budget = max(1, int(round(0.10 * len(cf.rows))))
    rows = []
    for m in cohort.spec.orderable:
        est = patient_expected_gain(cf, cohort, m, context=base,
                                    cfg=GainConfig(n_samples=24,
                                                   n_neighbors=40)).delta_bits
        retro = retrospective_gain(cf, m, context=base)
        true = gt.patient_gain(outcome, m, set(base), rows=cf.rows, n_mc=64,
                               seed=seed)
        # a budget policy buys the test for the `budget` highest-scoring
        # patients; measure the true information it thereby collects against
        # what the oracle ordering collects, with random ordering as the floor
        oracle = float(np.sort(true)[-budget:].sum())
        got = float(true[np.argsort(-est)[:budget]].sum())
        chance = float(true.mean() * budget)
        rows.append({
            "modality": m,
            "mean_est_bits": float(est.mean()),
            "mean_true_bits": float(true.mean()),
            "mean_retro_bits": float(retro.mean()),
            "spearman_est_vs_truth": float(spearmanr(est, true).statistic),
            "pearson_est_vs_truth": float(np.corrcoef(est, true)[0, 1]),
            "spearman_est_vs_retro": float(spearmanr(est, retro).statistic),
            "spearman_truth_vs_retro": float(spearmanr(true, retro).statistic),
            "capture_at_10": got / oracle if oracle > 0 else float("nan"),
            "chance_at_10": chance / oracle if oracle > 0 else float("nan"),
            "n": int(len(cf.rows)),
        })
    return pd.DataFrame(rows)


def main() -> None:  # pragma: no cover - CLI
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default="results/simulation")
    ap.add_argument("--outcome", default="mortality_30d")
    ap.add_argument("--sizes", default="4000,12000,36000")
    ap.add_argument("--qual-sizes", default="6000,18000,50000")
    ap.add_argument("--power-sizes", default="18000,50000")
    ap.add_argument("--exact-n", type=int, default=60000)
    ap.add_argument("--epochs", type=int, default=110)
    ap.add_argument("--folds", type=int, default=4)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rebuild-summary", action="store_true",
                    help="recompute summary.json from saved tables, no refitting")
    ap.add_argument("--targeting-n", type=int, default=50000,
                    help="cohort size for the per-patient targeting study")
    ap.add_argument("--targeting-only", action="store_true",
                    help="run only the per-patient targeting study and exit; it "
                         "needs one family fit rather than the whole grid")
    args = ap.parse_args()

    out = Path(args.out)
    if args.rebuild_summary:
        s = rebuild_summary(out, args.outcome)
        from infogain.utils.io import NumpyJSONEncoder

        print(json.dumps({k: v for k, v in s.items() if k != "regime_table"},
                         indent=2, cls=NumpyJSONEncoder))
        return
    (out / "tables").mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    sizes = tuple(int(s) for s in args.sizes.split(","))
    train = TrainConfig(epochs=args.epochs, n_folds=args.folds,
                        seeds=tuple(range(args.seeds)), patience=18)
    if args.quick:
        sizes = (2000, 5000, 10000)
        train = TrainConfig(epochs=25, n_folds=3, seeds=(0,), patience=8, min_epochs=8)
        args.qual_sizes, args.exact_n = "2000,5000", 15000
        args.power_sizes = "4000"
        args.epochs, args.folds, args.seeds = 25, 3, 1
        args.targeting_n = 4000

    if args.targeting_only:
        tgt = patient_targeting_study(args.targeting_n, args.outcome, args.epochs,
                                      args.folds, tuple(range(args.seeds)), args.seed)
        tgt.to_csv(out / "tables" / "patient_targeting.csv", index=False)
        print(tgt.round(4).to_string(index=False))
        return

    log.info("== recovery study ==")
    rec = add_extrapolation(recovery_study(sizes, args.outcome, args.seed, train))
    rec.to_csv(out / "tables" / "recovery.csv", index=False)

    log.info("== qualitative regime study ==")
    qual_sizes = tuple(int(x) for x in args.qual_sizes.split(","))
    qual = qualitative_study(qual_sizes, args.outcome, args.seed, train)
    qual.to_csv(out / "tables" / "regime_recovery.csv", index=False)
    regime_accuracy_by_size(qual).to_csv(out / "tables" / "regime_by_size.csv", index=False)
    first_correct_size(qual).to_csv(out / "tables" / "regime_first_correct.csv", index=False)

    log.info("== synergy power study ==")
    abl, ref = synergy_power(tuple(int(x) for x in args.power_sizes.split(",")),
                             (False, True), args.outcome, args.seed, train)
    abl.to_csv(out / "tables" / "synergy_power.csv", index=False)
    ref.to_csv(out / "tables" / "synergy_power_tree.csv", index=False)

    log.info("== per-patient targeting study ==")
    tgt = patient_targeting_study(args.targeting_n, args.outcome, args.epochs,
                                  args.folds, tuple(range(args.seeds)), args.seed)
    tgt.to_csv(out / "tables" / "patient_targeting.csv", index=False)

    log.info("== exact theorem checks ==")
    thm = theorem_study_exact(args.exact_n, args.outcome, args.seed)
    pd.DataFrame(thm["theorem1"]).to_csv(out / "tables" / "exact_theorem1.csv", index=False)
    pd.DataFrame(thm["theorem2"]).to_csv(out / "tables" / "exact_theorem2.csv", index=False)

    save(F.fig_recovery(rec, f"simulation: recovering known information ({args.outcome})"),
         out / "figures" / "figS1_recovery", table=rec)
    save(F.fig_regime_recovery(regime_accuracy_by_size(qual), first_correct_size(qual),
                               "simulation: recovering the decomposition's sign"),
         out / "figures" / "figS2_regime", table=qual)

    summary = summarize(rec, qual, thm, sizes, args.outcome, abl=abl, ref=ref)
    save_json(summary, out / "summary.json")
    print(rec.to_string(index=False))
    print("\n", qual.round(4).to_string(index=False))
    print(f"\nmedian recovery at n={max(sizes)} over the "
          f"{summary['n_subsets_scored']} subsets above "
          f"{summary['recovery_floor_bits']} bits: "
          f"{summary['recovery_at_largest_n']:.1%}"
          f"  (after learning-curve correction: "
          f"{summary['recovery_after_correction']:.1%})")
    print(regime_accuracy_by_size(qual).round(2).to_string(index=False))
    print(first_correct_size(qual).to_string(index=False))
    print(f"regime accuracy at n={summary['regime_max_n']}: "
          f"{summary['regime_accuracy']:.0%}   "
          f"Thm1 holds: {summary['theorem1_all_hold']}   "
          f"Thm2 holds: {summary['theorem2_all_hold']}   "
          f"Thm3 rel err: {thm['theorem3']['rel_error']:.2e}")


if __name__ == "__main__":  # pragma: no cover
    main()
