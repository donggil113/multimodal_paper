r"""Main INFOGAIN analysis: cohort in, every table and figure of the paper out.

Runs unchanged on a PhysioNet-derived cohort and on the simulator's output,
because both are :class:`~infogain.data.schema.Cohort` objects.  The stages are
the argument of the paper, in order:

1. fit the masked family by cross-fitting, and check it is worth trusting
   (calibration, clipping, monotonicity of the estimated lattice);
2. decompose each modality into marginal / conditional / redundant / synergistic
   value, and map the pairwise interactions;
3. attribute information across modalities by Shapley value, which removes the
   arbitrariness of "which test did you already have";
4. estimate the per-patient prospective gain for every orderable test, and check
   it against the retrospective realised gain;
5. translate bits into clinical currency -- Theorem 1's certified AUROC gain,
   Theorem 3's exact net-benefit identity, and the clinically restricted
   information that only counts bits redeemable at plausible thresholds;
6. simulate patient-level ordering policies against every comparator and report
   the reduction achievable at matched performance, with Theorem 2's certified
   bound on what that reduction can cost.
"""
from __future__ import annotations

import argparse
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from infogain.clinical.cost import CostModel, CostWeights
from infogain.clinical.net_benefit import (
    check_identity,
    empirical_net_benefit,
    nest_calibrate,
    restricted_information,
    safe_omission_bound_local,
)
from infogain.clinical.policy import reduction_study
from infogain.data.schema import Cohort
from infogain.encoders.fusion import FamilyConfig
from infogain.encoders.train import TrainConfig, fit_family
from infogain.theory.lemmas import achievable_auroc_bracket, auroc, best_stratified_bound
from infogain.utils.io import save_json
from infogain.utils.logging import get_logger
from infogain.vinfo.conditional import (
    GainConfig,
    overlap_diagnostics,
    patient_expected_gain,
    retrospective_gain,
)
from infogain.vinfo.decomposition import (
    decomposition_table, lattice_table, monotonicity_report,
    pairwise_interaction_map, shapley_information,
)
from infogain.vinfo.estimators import diagnostics_table, variance_decomposition
from infogain.viz import figures as F
from infogain.viz.style import save

log = get_logger("infogain.experiments.run_cohort")
LN2 = float(np.log(2.0))


@dataclass
class AnalysisConfig:
    outcome: str = "mortality_30d"
    train: TrainConfig = field(default_factory=TrainConfig)
    gain: GainConfig = field(default_factory=GainConfig)
    threshold_window: tuple[float, float] = (0.02, 0.30)
    decision_thresholds: tuple[float, ...] = (0.02, 0.05, 0.10, 0.20)
    cost_weights: CostWeights = field(default_factory=CostWeights)
    seed: int = 0
    make_figures: bool = True

    def describe(self) -> dict:
        d = {k: v for k, v in self.__dict__.items() if k not in ("train", "gain", "cost_weights")}
        d["train"] = asdict(self.train)
        d["gain"] = asdict(self.gain)
        d["cost_weights"] = asdict(self.cost_weights)
        return d


def analyse(cohort: Cohort, cfg: AnalysisConfig, out_dir: Path) -> dict:
    out_dir = Path(out_dir)
    (out_dir / "tables").mkdir(parents=True, exist_ok=True)
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    oc = cfg.outcome
    baseline = cohort.spec.baseline
    orderable = [m for m in cohort.spec.names if m not in baseline]
    summary: dict = {"outcome": oc, "config": cfg.describe(),
                     "cohort": {"n": cohort.n,
                                "n_subjects": int(cohort.index["subject_id"].nunique()),
                                "prevalence": cohort.prevalence(oc),
                                "baseline": sorted(baseline),
                                "orderable": orderable,
                                "source": cohort.meta.get("source", "unknown")}}

    # ---- 1. fit -------------------------------------------------------------
    log.info("[1/6] cross-fitting the masked family")
    cf = fit_family(cohort, oc, cfg.train, keep_models=True)
    y = cf.y
    prev = float(y.mean())
    summary["null_gap_bits"] = cf.null_gap_bits()

    diag = diagnostics_table(cf)
    diag.to_csv(out_dir / "tables" / "diagnostics.csv", index=False)
    lat = lattice_table(cf)
    lat.to_csv(out_dir / "tables" / "lattice.csv", index=False)
    mono = monotonicity_report(cf)
    summary["monotonicity"] = {k: v for k, v in mono.items()
                               if k not in ("raw", "projected")}
    full = frozenset(cohort.spec.names)
    vd = variance_decomposition(cf.pvi_all_seeds(full))
    summary["variance_decomposition_full_panel"] = vd.as_dict()
    summary["information"] = {
        "baseline_bits": float(cf.pvi(baseline).mean()),
        "full_panel_bits": float(cf.pvi(full).mean()),
        "full_panel_auroc": float(auroc(y, cf.p(full))),
        "baseline_auroc": float(auroc(y, cf.p(baseline))),
    }

    # ---- 2. decomposition ---------------------------------------------------
    log.info("[2/6] decomposing")
    dec_loo = decomposition_table(cf)                       # leave-one-out context
    dec_loo["context_kind"] = "leave_one_out"
    dec_base = decomposition_table(cf, context=baseline)    # nothing else ordered yet
    dec_base["context_kind"] = "baseline_only"
    dec = pd.concat([dec_loo, dec_base], ignore_index=True)
    dec.to_csv(out_dir / "tables" / "decomposition.csv", index=False)

    inter = pairwise_interaction_map(cf)
    inter.to_csv(out_dir / "tables" / "interaction_map.csv", index=False)

    shap = shapley_information(cf)
    shap.to_csv(out_dir / "tables" / "shapley.csv", index=False)
    summary["shapley_total_bits"] = float(shap.attrs["total_bits"])
    summary["shapley_efficiency_residual"] = float(shap.attrs["efficiency_residual"])

    # ---- 3. per-patient gains ----------------------------------------------
    log.info("[3/6] per-patient prospective gains")
    gains: dict[str, np.ndarray] = {}
    gain_rows, consistency, overlap = [], [], []
    for m in orderable:
        res = patient_expected_gain(cf, cohort, m, context=baseline, cfg=cfg.gain)
        gains[m] = res.delta_bits
        gain_rows.append(res.summary())
        retro = retrospective_gain(cf, m, context=baseline)
        consistency.append({
            "modality": m,
            "prospective_mean_bits": float(res.delta_bits.mean()),
            "retrospective_mean_bits": float(retro.mean()),
            "abs_difference": float(abs(res.delta_bits.mean() - retro.mean())),
            "spearman": float(pd.Series(res.delta_bits).corr(pd.Series(retro),
                                                             method="spearman")),
        })
        overlap.append(overlap_diagnostics(cohort, m, sorted(baseline), rows=cf.rows))
    pd.DataFrame(gain_rows).to_csv(out_dir / "tables" / "patient_gain_summary.csv", index=False)
    pd.DataFrame(consistency).to_csv(out_dir / "tables" / "gain_consistency.csv", index=False)
    pd.DataFrame(overlap).to_csv(out_dir / "tables" / "overlap_diagnostics.csv", index=False)
    np.savez_compressed(out_dir / "patient_gains.npz",
                        **{m: g for m, g in gains.items()},
                        base_risk=cf.p(baseline), y=y)
    summary["patient_gains"] = gain_rows
    summary["gain_consistency"] = consistency
    summary["overlap"] = overlap

    # ---- 4. clinical translation -------------------------------------------
    log.info("[4/6] translating bits into clinical currency")
    p_base = cf.p(baseline)
    p_full = cf.p(full)
    p_base_nested = nest_calibrate(p_full, p_base)
    ident = check_identity(p_full, p_base_nested)
    lo_t, hi_t = cfg.threshold_window
    summary["identity_check"] = {
        "kl_bits": ident.kl_nats / LN2, "integral_bits": ident.integral_nats / LN2,
        "rel_error": ident.rel_error, "passes": bool(ident.passes),
        "restricted_bits": restricted_information(p_full, p_base_nested, lo_t, hi_t),
        "window": [lo_t, hi_t]}

    thm1_rows = []
    for m in orderable:
        ctx = full - {m}
        b = best_stratified_bound(y, cf.p(ctx), cf.p(full))
        lo_b, hi_b = achievable_auroc_bracket(
            float((cf.pvi(full) - cf.pvi(ctx)).mean()), prev)
        thm1_rows.append({"modality": m, "bound": b.bound,
                          "observed_gain": b.observed_gain,
                          "achievable_gain": b.achievable_gain,
                          "auroc_context": b.auroc_coarse, "auroc_full": b.auroc_fine,
                          "n_bins": b.n_bins,
                          "info_only_auroc_floor": lo_b - 0.5,
                          "info_only_auroc_ceiling": hi_b - 0.5})
    thm1 = pd.DataFrame(thm1_rows)
    thm1.to_csv(out_dir / "tables" / "theorem1_bounds.csv", index=False)

    thm2_rows = []
    for m in orderable:
        ctx = full - {m}
        forgone = max(float((cf.pvi(full) - cf.pvi(ctx)).mean()), 0.0)
        for t in cfg.decision_thresholds:
            thm2_rows.append({"modality": m, "threshold": t,
                              "forgone_bits": forgone,
                              "certified_nb_loss": safe_omission_bound_local(forgone, t),
                              # the information component, the one Theorem 2
                              # bounds: evaluated between the full model and its
                              # projection onto the reduced model's ranking
                              "observed_nb_loss": float(
                                  empirical_net_benefit(y, cf.p(full), np.array([t]))[0]
                                  - empirical_net_benefit(
                                      y, nest_calibrate(cf.p(full), cf.p(ctx)),
                                      np.array([t]))[0]),
                              "deployed_nb_loss": float(
                                  empirical_net_benefit(y, cf.p(full), np.array([t]))[0]
                                  - empirical_net_benefit(y, cf.p(ctx), np.array([t]))[0])})
    pd.DataFrame(thm2_rows).to_csv(out_dir / "tables" / "theorem2_bounds.csv", index=False)

    # ---- 5. reduction study -------------------------------------------------
    log.info("[5/6] simulating ordering policies")
    cost_model = CostModel(cohort.spec, cfg.cost_weights)
    cost_model.table().to_csv(out_dir / "tables" / "cost_model.csv", index=False)
    study = reduction_study(cf, cohort, gains, cost_model,
                            thresholds=cfg.decision_thresholds, seed=cfg.seed)
    study.frontier.to_csv(out_dir / "tables" / "policy_frontier.csv", index=False)
    study.comparators.to_csv(out_dir / "tables" / "policy_comparators.csv", index=False)
    summary["reduction"] = study.equal_performance
    summary["reduction_summary_text"] = study.summary()
    log.info(study.summary())

    # ---- 6. figures ---------------------------------------------------------
    if cfg.make_figures:
        log.info("[6/6] rendering figures")
        fd = out_dir / "figures"
        save(F.fig_decomposition(dec_loo, f"{oc}: value of each test given all others"),
             fd / "fig1_decomposition", table=dec_loo)
        save(F.fig_interaction_map(inter, orderable,
                                   f"{oc}: modality-pair interaction"),
             fd / "fig2_interaction_map", table=inter)
        save(F.fig_patient_gains(gains, f"{oc}: who benefits from testing"),
             fd / "fig3_patient_gains",
             table=pd.DataFrame({m: g for m, g in gains.items()}))
        ts = np.linspace(0.005, 0.5, 200)
        curves = {"order_all": (ts, empirical_net_benefit(y, p_full, ts)),
                  "baseline_only": (ts, empirical_net_benefit(y, p_base, ts))}
        best_lam = study.equal_performance.get("lambda")
        if best_lam is not None:
            from infogain.clinical.policy import policy_infogain

            pol = policy_infogain(cf, cost_model, gains, best_lam)
            curves["infogain"] = (ts, empirical_net_benefit(y, pol.probs, ts))
        save(F.fig_decision_curves(curves, prev, ts, f"{oc}: decision curves"),
             fd / "fig4_decision_curves",
             table=pd.DataFrame({"threshold": ts,
                                 **{k: v[1] for k, v in curves.items()}}))
        save(F.fig_reduction_frontier(study.frontier, study.comparators, "auroc",
                                      f"{oc}: tests ordered vs discrimination"),
             fd / "fig5_reduction_frontier", table=study.frontier)
        d_nb = (empirical_net_benefit(y, p_full, ts)
                - empirical_net_benefit(y, p_base, ts))
        save(F.fig_identity(ts, d_nb, ident.kl_nats / LN2, ident.integral_nats / LN2,
                            cfg.threshold_window, f"{oc}: Theorem 3"),
             fd / "fig7_identity",
             table=pd.DataFrame({"threshold": ts, "delta_net_benefit": d_nb}))
        save(F.fig_theorem1(thm1, f"{oc}: Theorem 1 certified AUROC gain"),
             fd / "fig8_theorem1", table=thm1)
        save(F.fig_gain_by_risk(p_base, gains, title=f"{oc}: gain vs baseline risk"),
             fd / "fig9_gain_by_risk")
        save(F.fig_calibration(y, {"baseline": p_base, "full panel": p_full},
                               title=f"{oc}: calibration"),
             fd / "fig10_calibration")

    summary["runtime_seconds"] = time.time() - t_start
    save_json(summary, out_dir / "summary.json")
    log.info("done in %.0fs -> %s", summary["runtime_seconds"], out_dir)
    return summary


def _quick(cfg: AnalysisConfig) -> AnalysisConfig:
    cfg.train = TrainConfig(epochs=25, n_folds=3, seeds=(0,), patience=8,
                            min_epochs=8, family=FamilyConfig())
    cfg.gain = GainConfig(n_samples=8, n_neighbors=20)
    return cfg


def main() -> None:  # pragma: no cover - CLI
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cohort", required=True, help="directory written by Cohort.save")
    ap.add_argument("--outcome", default="mortality_30d")
    ap.add_argument("--out", default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--folds", type=int, default=None)
    ap.add_argument("--seeds", type=int, default=None, help="number of seeds")
    ap.add_argument("--gain-samples", type=int, default=None)
    ap.add_argument("--cost-mode", default="money",
                    choices=["money", "throughput", "harm"])
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--no-figures", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cohort = Cohort.load(args.cohort)
    cfg = AnalysisConfig(outcome=args.outcome, seed=args.seed,
                         make_figures=not args.no_figures)
    if args.quick:
        cfg = _quick(cfg)
    if args.epochs:
        cfg.train.epochs = args.epochs
    if args.folds:
        cfg.train.n_folds = args.folds
    if args.seeds:
        cfg.train.seeds = tuple(range(args.seeds))
    if args.gain_samples:
        cfg.gain.n_samples = args.gain_samples
    cfg.cost_weights = {"money": CostWeights.money_only(),
                        "throughput": CostWeights.throughput_limited(),
                        "harm": CostWeights.harm_averse()}[args.cost_mode]

    out = Path(args.out or f"results/{Path(args.cohort).name}/{args.outcome}")
    print(cohort.describe())
    summary = analyse(cohort, cfg, out)
    print("\n" + summary.get("reduction_summary_text", ""))


if __name__ == "__main__":  # pragma: no cover
    main()
