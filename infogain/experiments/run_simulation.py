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
import time
from pathlib import Path

import numpy as np
import pandas as pd

from infogain.clinical.net_benefit import (
    bayes_value, check_identity, information_gain_nats, safe_omission_bound_local,
)
from infogain.data.synthetic import generate, ground_truth_table
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


def qualitative_study(n: int = 24000, outcome: str = "mortality_30d", seed: int = 0,
                      train: TrainConfig | None = None) -> pd.DataFrame:
    """Does the estimated decomposition agree with the known regime per modality?"""
    train = train or TrainConfig(epochs=110, n_folds=4, seeds=(0, 1), patience=18)
    cohort, gt = generate(n=n, seed=seed)
    cf = fit_family(cohort, outcome, train)
    est = decomposition_table(cf).set_index("modality")
    truth = ground_truth_table(gt, outcome, baseline=sorted(cohort.spec.baseline),
                               respect_observation=True).set_index("modality")

    def regime(marg: float, cond: float, tol: float = 0.002) -> str:
        if cond - marg > tol:
            return "synergistic"
        if marg - cond > tol:
            return "redundant"
        return "independent"

    rows = []
    for m in est.index:
        if m not in truth.index:
            continue
        t, e = truth.loc[m], est.loc[m]
        rows.append({
            "modality": m,
            "true_marginal": t["marginal"], "est_marginal": e["marginal_bits"],
            "true_conditional": t["conditional"], "est_conditional": e["conditional_bits"],
            "true_redundant": t["redundant"], "est_redundant": e["redundant_bits"],
            "true_synergistic": t["synergistic"], "est_synergistic": e["synergistic_bits"],
            "true_regime": regime(t["marginal"], t["conditional"]),
            "est_regime": regime(e["marginal_bits"], e["conditional_bits"]),
        })
    out = pd.DataFrame(rows)
    out["regime_correct"] = out["true_regime"] == out["est_regime"]
    return out


def theorem_study_exact(n: int = 60000, outcome: str = "mortality_30d",
                        seed: int = 0) -> dict:
    """Check Theorems 1-3 against *exact* posteriors from the simulator."""
    cohort, gt = generate(n=n, seed=seed)
    y = gt.y[outcome]
    base = sorted(cohort.spec.baseline)
    full = set(cohort.spec.modalities)
    p_full = gt.posterior(outcome, full, respect_observation=False)

    thm1, thm2 = [], []
    for m in (set(cohort.spec.names) - set(base)):
        ctx = full - {m}
        p_ctx = gt.posterior(outcome, ctx, respect_observation=False)
        b = best_stratified_bound(y, p_ctx, p_full)
        thm1.append({"modality": m, "bound": b.bound,
                     "achievable_gain": auroc_hull(y, p_full) - auroc(y, p_ctx),
                     "observed_gain": b.observed_gain,
                     "holds": bool(b.bound <= auroc_hull(y, p_full) - auroc(y, p_ctx) + 1e-9)})
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


def main() -> None:  # pragma: no cover - CLI
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default="results/simulation")
    ap.add_argument("--outcome", default="mortality_30d")
    ap.add_argument("--sizes", default="4000,12000,36000")
    ap.add_argument("--qual-n", type=int, default=24000)
    ap.add_argument("--exact-n", type=int, default=60000)
    ap.add_argument("--epochs", type=int, default=110)
    ap.add_argument("--folds", type=int, default=4)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out = Path(args.out)
    (out / "tables").mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    sizes = tuple(int(s) for s in args.sizes.split(","))
    train = TrainConfig(epochs=args.epochs, n_folds=args.folds,
                        seeds=tuple(range(args.seeds)), patience=18)
    if args.quick:
        sizes = (2000, 5000, 10000)
        train = TrainConfig(epochs=25, n_folds=3, seeds=(0,), patience=8, min_epochs=8)
        args.qual_n, args.exact_n = 6000, 15000

    log.info("== recovery study ==")
    rec = add_extrapolation(recovery_study(sizes, args.outcome, args.seed, train))
    rec.to_csv(out / "tables" / "recovery.csv", index=False)

    log.info("== qualitative regime study ==")
    qual = qualitative_study(args.qual_n, args.outcome, args.seed, train)
    qual.to_csv(out / "tables" / "regime_recovery.csv", index=False)

    log.info("== exact theorem checks ==")
    thm = theorem_study_exact(args.exact_n, args.outcome, args.seed)
    pd.DataFrame(thm["theorem1"]).to_csv(out / "tables" / "exact_theorem1.csv", index=False)
    pd.DataFrame(thm["theorem2"]).to_csv(out / "tables" / "exact_theorem2.csv", index=False)

    save(F.fig_recovery(rec, f"simulation: recovering known information ({args.outcome})"),
         out / "figures" / "figS1_recovery", table=rec)

    summary = {
        "sizes": list(sizes), "outcome": args.outcome,
        "recovery_at_largest_n": float(
            rec[rec["n"] == max(sizes)]["recovered"].median()),
        "recovery_after_correction": float(
            (rec[rec["n"] == max(sizes)]["corrected_bits"]
             / rec[rec["n"] == max(sizes)]["truth_bits"].clip(lower=1e-9)).median()),
        "regime_accuracy": float(qual["regime_correct"].mean()),
        "regime_table": qual.to_dict("records"),
        "theorem1_all_hold": bool(all(r["holds"] for r in thm["theorem1"])),
        "theorem2_all_hold": bool(all(r["holds"] for r in thm["theorem2"])),
        "theorem3": thm["theorem3"],
    }
    save_json(summary, out / "summary.json")
    print(rec.to_string(index=False))
    print("\n", qual.round(4).to_string(index=False))
    print(f"\nmedian recovery at n={max(sizes)}: {summary['recovery_at_largest_n']:.1%}"
          f"  (after learning-curve correction: {summary['recovery_after_correction']:.1%})")
    print(f"regime accuracy: {summary['regime_accuracy']:.0%}   "
          f"Thm1 holds: {summary['theorem1_all_hold']}   "
          f"Thm2 holds: {summary['theorem2_all_hold']}   "
          f"Thm3 rel err: {thm['theorem3']['rel_error']:.2e}")


if __name__ == "__main__":  # pragma: no cover
    main()
