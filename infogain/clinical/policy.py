r"""Patient-level test-ordering policies and the retrospective reduction study.

Given per-patient gains :math:`\Delta_i(m\mid S)` (:mod:`infogain.vinfo.conditional`)
and a cost model, a policy assigns each patient a personalised panel
:math:`S_i`.  Its value is then evaluated **honestly**: the prediction for
patient :math:`i` uses only the modalities the policy actually ordered, read off
the cross-fitted probability for that exact subset.  No policy is ever scored
with information it declined to buy.

Comparators are chosen so the headline claim cannot be won cheaply:

``order_all``          today's maximal-testing practice: the performance ceiling
``baseline_only``      order nothing beyond what is free: the cost floor
``fixed:<m>``          always order one specific test
``random``             random ordering matched to the same budget
``risk_band``          the standard clinical heuristic -- test the intermediate-risk
``infogain``           order iff :math:`\Delta_i(m)/c_m` exceeds a threshold

A reduction claim is only interesting against ``random`` and ``risk_band`` at
*matched budget*; beating ``baseline_only`` is trivial and beating ``order_all``
on cost is tautological.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from infogain.clinical.cost import CostModel
from infogain.clinical.net_benefit import (
    empirical_net_benefit, information_gain_nats, nest_calibrate,
    restricted_information, safe_omission_bound_local,
)
from infogain.theory.lemmas import auroc
from infogain.utils.logging import get_logger
from infogain.vinfo.core import subset_key

log = get_logger("infogain.clinical.policy")
LN2 = float(np.log(2.0))


# --------------------------------------------------------------------------- #
# Policies
# --------------------------------------------------------------------------- #
@dataclass
class PolicyResult:
    name: str
    chosen: dict[str, np.ndarray]      # modality -> (n,) bool "ordered"
    probs: np.ndarray                  # (n,) prediction under the chosen panel
    cost: np.ndarray                   # (n,) cost incurred
    param: float = float("nan")

    @property
    def n_tests(self) -> np.ndarray:
        return np.sum([v.astype(int) for v in self.chosen.values()], axis=0)


def _subset_for_patient(baseline: frozenset[str], chosen: dict[str, np.ndarray],
                        i: int) -> frozenset[str]:
    return baseline | frozenset(m for m, v in chosen.items() if v[i])


def probs_under_policy(cf, baseline: frozenset[str],
                       chosen: dict[str, np.ndarray]) -> np.ndarray:
    """Read each patient's prediction off the subset the policy actually bought."""
    n = len(cf.y)
    out = np.empty(n, dtype=np.float64)
    orderable = sorted(chosen)
    codes = np.zeros(n, dtype=np.int64)
    for b, m in enumerate(orderable):
        codes |= (chosen[m].astype(np.int64) << b)
    for code in np.unique(codes):
        sel = codes == code
        s = baseline | frozenset(m for b, m in enumerate(orderable) if (code >> b) & 1)
        if not cf.has(s):
            raise KeyError(f"policy requires unfitted subset {subset_key(s)}")
        out[sel] = cf.p(s)[sel]
    return out


def policy_order_all(cf, cost_model: CostModel, orderable: Sequence[str]) -> PolicyResult:
    n = len(cf.y)
    chosen = {m: np.ones(n, bool) for m in orderable}
    cost = np.full(n, sum(cost_model.cost(m) for m in orderable))
    return PolicyResult("order_all", chosen,
                        probs_under_policy(cf, cf.baseline, chosen), cost)


def policy_baseline_only(cf, cost_model: CostModel,
                         orderable: Sequence[str]) -> PolicyResult:
    n = len(cf.y)
    chosen = {m: np.zeros(n, bool) for m in orderable}
    return PolicyResult("baseline_only", chosen,
                        probs_under_policy(cf, cf.baseline, chosen), np.zeros(n))


def policy_fixed(cf, cost_model: CostModel, orderable: Sequence[str],
                 keep: Iterable[str]) -> PolicyResult:
    n = len(cf.y)
    keep = set(keep)
    chosen = {m: np.full(n, m in keep) for m in orderable}
    cost = np.full(n, sum(cost_model.cost(m) for m in keep))
    return PolicyResult(f"fixed:{'+'.join(sorted(keep)) or 'none'}", chosen,
                        probs_under_policy(cf, cf.baseline, chosen), cost)


def policy_infogain(cf, cost_model: CostModel, deltas: dict[str, np.ndarray],
                    threshold: float, per_dollar: bool = True) -> PolicyResult:
    r"""Order :math:`m` for patient :math:`i` iff its yield clears ``threshold``.

    With ``per_dollar`` the criterion is :math:`\Delta_i(m)/c_m > \lambda`
    (bits per dollar), which is the Lagrangian solution to maximising total
    information under a budget; without it, the criterion is an absolute
    information floor, which is the right form when the constraint is "don't
    skip anything that could matter" rather than money.
    """
    n = len(cf.y)
    chosen, cost = {}, np.zeros(n)
    for m, d in deltas.items():
        c = cost_model.cost(m)
        score = d / max(c, 1e-9) if per_dollar else d
        take = score > threshold
        chosen[m] = take
        cost += take * c
    return PolicyResult("infogain", chosen,
                        probs_under_policy(cf, cf.baseline, chosen), cost,
                        param=threshold)


def policy_random(cf, cost_model: CostModel, orderable: Sequence[str],
                  rate: float, seed: int = 0) -> PolicyResult:
    n = len(cf.y)
    rng = np.random.default_rng(seed)
    chosen, cost = {}, np.zeros(n)
    for m in orderable:
        take = rng.random(n) < rate
        chosen[m] = take
        cost += take * cost_model.cost(m)
    return PolicyResult("random", chosen,
                        probs_under_policy(cf, cf.baseline, chosen), cost, param=rate)


def policy_risk_band(cf, cost_model: CostModel, orderable: Sequence[str],
                     base_risk: np.ndarray, lo_q: float, hi_q: float) -> PolicyResult:
    """Test the diagnostically uncertain middle -- the standard clinical heuristic.

    Patients whose baseline risk is already very low or very high are left
    untested on the grounds that no result would change management.  This is the
    comparator to beat: it is what a thoughtful clinician does without any
    information theory, and it is genuinely good.
    """
    n = len(cf.y)
    lo, hi = np.quantile(base_risk, [lo_q, hi_q])
    take = (base_risk >= lo) & (base_risk <= hi)
    chosen = {m: take.copy() for m in orderable}
    cost = take * sum(cost_model.cost(m) for m in orderable)
    return PolicyResult(f"risk_band[{lo_q:.2f},{hi_q:.2f}]", chosen,
                        probs_under_policy(cf, cf.baseline, chosen), cost,
                        param=hi_q - lo_q)


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
@dataclass
class PolicyEvaluation:
    name: str
    param: float
    tests_per_patient: float
    test_fraction: float
    mean_cost: float
    auroc: float
    net_benefit: dict[float, float]           # as deployed, uncalibrated
    net_benefit_nested: dict[float, float]    # after the nesting projection
    restricted_info_bits: float
    info_vs_all_bits: float
    certified_nb_loss: dict[float, float]
    info_nb_loss: dict[float, float]          # the part Theorem 2 certifies
    calibration_nb_loss: dict[float, float]   # the part recalibration removes
    missed_cases_delta: dict[float, int]

    def as_row(self, thresholds: Sequence[float]) -> dict:
        row = {"policy": self.name, "param": self.param,
               "tests_per_patient": self.tests_per_patient,
               "test_fraction": self.test_fraction, "mean_cost": self.mean_cost,
               "auroc": self.auroc,
               "restricted_info_bits": self.restricted_info_bits,
               "info_gap_vs_all_bits": self.info_vs_all_bits}
        for t in thresholds:
            row[f"nb@{t:g}"] = self.net_benefit.get(t, np.nan)
            row[f"nb_loss_bound@{t:g}"] = self.certified_nb_loss.get(t, np.nan)
            row[f"nb_loss_info@{t:g}"] = self.info_nb_loss.get(t, np.nan)
            row[f"nb_loss_calib@{t:g}"] = self.calibration_nb_loss.get(t, np.nan)
            row[f"missed@{t:g}"] = self.missed_cases_delta.get(t, np.nan)
        return row


def evaluate_policy(res: PolicyResult, y: np.ndarray, p_reference: np.ndarray,
                    thresholds: Sequence[float] = (0.02, 0.05, 0.10, 0.20),
                    n_orderable: int = 1,
                    restricted_window: tuple[float, float] = (0.02, 0.30)
                    ) -> PolicyEvaluation:
    r"""Score a policy against the order-everything reference.

    A reduced policy loses net benefit for two separable reasons, and conflating
    them is a mistake we take some care to avoid.

    *Information.*  It has fewer inputs, so its posterior is coarser.  This is
    what ``info_vs_all_bits`` measures and what Theorem 2 certifies: the loss is
    evaluated between the reference and its projection onto the policy's
    ranking, the nested pair the theorem is stated for.

    *Calibration.*  A model with fewer inputs is typically also mis-calibrated
    relative to the reference, so thresholding it selects a different set even
    where the ranking agrees.  That loss is real as deployed, but it is removed
    by recalibration rather than by ordering more tests, and Theorem 2 does not
    cover it.  ``calibration_nb_loss`` reports it separately.

    A monotone recalibration of the full model therefore shows an information
    gap of exactly zero -- as it should, since it has the same information.
    """
    p = res.probs
    nested = nest_calibrate(p_reference, p)
    gap_bits = information_gain_nats(p_reference, nested) / LN2
    nb, nb_nested, missed, cert, info_loss, calib_loss = {}, {}, {}, {}, {}, {}
    for t in thresholds:
        tt = np.array([t])
        nb_ref = float(empirical_net_benefit(y, p_reference, tt)[0])
        nb[t] = float(empirical_net_benefit(y, p, tt)[0])
        nb_nested[t] = float(empirical_net_benefit(y, nested, tt)[0])
        info_loss[t] = nb_ref - nb_nested[t]
        calib_loss[t] = nb_nested[t] - nb[t]
        ref_flag = p_reference > t
        pol_flag = p > t
        missed[t] = int(np.sum((y == 1) & ref_flag & ~pol_flag)
                        - np.sum((y == 1) & ~ref_flag & pol_flag))
        cert[t] = safe_omission_bound_local(gap_bits, t)
    return PolicyEvaluation(
        name=res.name, param=res.param,
        tests_per_patient=float(res.n_tests.mean()),
        test_fraction=float(res.n_tests.mean() / max(n_orderable, 1)),
        mean_cost=float(res.cost.mean()), auroc=float(auroc(y, p)),
        net_benefit=nb, net_benefit_nested=nb_nested,
        info_nb_loss=info_loss, calibration_nb_loss=calib_loss,
        # The identity behind restricted_information needs a *nested* coarse
        # model. Against the no-information reference the coarse posterior is a
        # constant, and nesting then requires that constant to be the mean of
        # the fine model, not the observed prevalence -- they differ by the
        # model's calibration error, which would otherwise leak into the number.
        restricted_info_bits=restricted_information(p, np.full_like(p, float(p.mean())),
                                                    *restricted_window),
        info_vs_all_bits=gap_bits, certified_nb_loss=cert,
        missed_cases_delta=missed)


@dataclass
class ReductionStudy:
    frontier: pd.DataFrame
    comparators: pd.DataFrame
    thresholds: tuple[float, ...]
    equal_performance: dict = field(default_factory=dict)

    def summary(self) -> str:
        eq = self.equal_performance
        if not eq:
            return "no equal-performance point found"
        return (f"At matched AUROC (within {eq['tolerance']:.3f}), INFOGAIN uses "
                f"{eq['test_fraction']:.1%} of tests vs 100% for order-all: "
                f"{eq['tests_avoided']:.1%} avoided, "
                f"${eq['cost_saved']:.2f}/patient saved, "
                f"AUROC {eq['auroc']:.4f} vs {eq['auroc_all']:.4f}, "
                f"certified net-benefit loss <= {eq['max_certified_nb_loss']:.5f}")


def reduction_study(cf, cohort, deltas: dict[str, np.ndarray],
                    cost_model: CostModel,
                    thresholds: Sequence[float] = (0.02, 0.05, 0.10, 0.20),
                    n_lambda: int = 25, seed: int = 0,
                    auroc_tolerance: float = 0.005) -> ReductionStudy:
    """Sweep the INFOGAIN threshold and compare against every baseline policy."""
    y = cf.y
    orderable = sorted(deltas)
    all_res = policy_order_all(cf, cost_model, orderable)
    p_ref = all_res.probs
    base_res = policy_baseline_only(cf, cost_model, orderable)

    scores = np.concatenate([deltas[m] / max(cost_model.cost(m), 1e-9)
                             for m in orderable])
    scores = scores[np.isfinite(scores) & (scores > 0)]
    if scores.size == 0:
        lambdas = np.array([0.0])
    else:
        qs = np.linspace(0.0, 0.995, n_lambda)
        lambdas = np.unique(np.concatenate([[0.0], np.quantile(scores, qs)]))

    rows = []
    for lam in lambdas:
        res = policy_infogain(cf, cost_model, deltas, lam, per_dollar=True)
        ev = evaluate_policy(res, y, p_ref, thresholds, len(orderable))
        rows.append(ev.as_row(thresholds))
    frontier = pd.DataFrame(rows).sort_values("test_fraction").reset_index(drop=True)

    comps = [evaluate_policy(all_res, y, p_ref, thresholds, len(orderable)).as_row(thresholds),
             evaluate_policy(base_res, y, p_ref, thresholds, len(orderable)).as_row(thresholds)]
    for m in orderable:
        r = policy_fixed(cf, cost_model, orderable, {m})
        comps.append(evaluate_policy(r, y, p_ref, thresholds, len(orderable)).as_row(thresholds))
    base_risk = cf.p(cf.baseline)
    for lo_q, hi_q in ((0.4, 0.9), (0.25, 0.95), (0.5, 0.95), (0.6, 1.0)):
        r = policy_risk_band(cf, cost_model, orderable, base_risk, lo_q, hi_q)
        comps.append(evaluate_policy(r, y, p_ref, thresholds, len(orderable)).as_row(thresholds))
    for rate in (0.25, 0.5, 0.75):
        r = policy_random(cf, cost_model, orderable, rate, seed)
        comps.append(evaluate_policy(r, y, p_ref, thresholds, len(orderable)).as_row(thresholds))
    comparators = pd.DataFrame(comps)

    auroc_all = float(comparators.loc[comparators["policy"] == "order_all", "auroc"].iloc[0])
    cost_all = float(comparators.loc[comparators["policy"] == "order_all", "mean_cost"].iloc[0])
    ok = frontier[frontier["auroc"] >= auroc_all - auroc_tolerance]
    eq = {}
    if len(ok):
        best = ok.sort_values("test_fraction").iloc[0]
        nb_cols = [c for c in frontier.columns if c.startswith("nb_loss_bound@")]
        eq = {"tolerance": auroc_tolerance,
              "lambda": float(best["param"]),
              "test_fraction": float(best["test_fraction"]),
              "tests_avoided": float(1.0 - best["test_fraction"]),
              "cost_saved": float(cost_all - best["mean_cost"]),
              "cost_saved_pct": float(1.0 - best["mean_cost"] / max(cost_all, 1e-9)),
              "auroc": float(best["auroc"]), "auroc_all": auroc_all,
              "info_gap_bits": float(best["info_gap_vs_all_bits"]),
              "max_certified_nb_loss": float(max(best[c] for c in nb_cols))}
    return ReductionStudy(frontier=frontier, comparators=comparators,
                          thresholds=tuple(thresholds), equal_performance=eq)


# --------------------------------------------------------------------------- #
# Sequential (greedy) acquisition
# --------------------------------------------------------------------------- #
def greedy_sequential(cf, gain_by_context: dict[tuple[str, ...], dict[str, np.ndarray]],
                      cost_model: CostModel, orderable: Sequence[str],
                      max_tests: int = 2,
                      lambda_bits_per_dollar: float = 0.0) -> PolicyResult:
    r"""Acquire tests one at a time, re-scoring against what has already arrived.

    ``gain_by_context`` maps a sorted tuple of already-acquired modalities to
    ``{modality: (n,) bits}`` -- the per-patient gain of each remaining test
    given that context.  The caller precomputes it (see
    :func:`sequential_gain_table`), because each entry costs a pass of the
    conditional sampler and only the caller knows the compute budget.

    Re-scoring matters exactly when modalities are redundant.  A one-shot policy
    scores every test against the free baseline, so for a patient whose chest
    radiograph and laboratory panel would say the same thing it buys both: each
    looks valuable on its own. After the radiograph shows congestion, the panel's
    remaining value for *that patient* collapses, and only a policy that looks
    again can see it.
    """
    n = len(cf.y)
    chosen = {m: np.zeros(n, bool) for m in orderable}
    cost = np.zeros(n)
    active = np.ones(n, bool)

    for step in range(max_tests):
        best_score = np.full(n, -np.inf)
        best_mod = np.full(n, -1)
        # group patients by the context they have reached, so each patient is
        # scored against their own acquired set rather than a cohort-level one
        codes = np.zeros(n, dtype=np.int64)
        for b, m in enumerate(sorted(orderable)):
            codes |= (chosen[m].astype(np.int64) << b)
        for code in np.unique(codes[active]):
            sel = active & (codes == code)
            ctx = tuple(sorted(m for b, m in enumerate(sorted(orderable))
                               if (code >> b) & 1))
            gains = gain_by_context.get(ctx)
            if gains is None:
                log.warning("no precomputed gains for context %s; skipping", ctx or "()")
                continue
            for j, m in enumerate(orderable):
                if m not in gains or chosen[m][sel].all():
                    continue
                score = gains[m] / max(cost_model.cost(m), 1e-9)
                upd = sel & ~chosen[m] & (score > best_score) & (score > lambda_bits_per_dollar)
                best_score = np.where(upd, score, best_score)
                best_mod = np.where(upd, j, best_mod)

        take = best_mod >= 0
        if not take.any():
            break
        for j, m in enumerate(orderable):
            pick = take & (best_mod == j)
            chosen[m] |= pick
            cost += pick * cost_model.cost(m)
        active &= take
        log.info("greedy step %d: ordered for %d patients", step + 1, int(take.sum()))

    return PolicyResult(f"greedy(max={max_tests})", chosen,
                        probs_under_policy(cf, cf.baseline, chosen), cost)


def sequential_gain_table(cf, cohort, orderable: Sequence[str], gain_cfg,
                          max_tests: int = 2) -> dict[tuple[str, ...], dict[str, np.ndarray]]:
    r"""Precompute per-patient gains for every context a greedy policy can reach.

    With :math:`k` orderable modalities and two steps that is
    :math:`k + k(k-1)` conditional-sampler passes, which is why the sequential
    policy is opt-in: it costs roughly :math:`k` times a one-shot run.
    """
    from itertools import combinations

    from infogain.vinfo.conditional import patient_expected_gain

    base = cf.baseline
    out: dict[tuple[str, ...], dict[str, np.ndarray]] = {}
    for r in range(max_tests):
        for acquired in combinations(sorted(orderable), r):
            ctx = base | frozenset(acquired)
            gains = {}
            for m in orderable:
                if m in acquired:
                    continue
                if not cf.has(ctx | {m}):
                    continue
                gains[m] = patient_expected_gain(cf, cohort, m, context=ctx,
                                                 cfg=gain_cfg).delta_bits
            if gains:
                out[tuple(sorted(acquired))] = gains
            log.info("sequential gains for context %s: %d modalities",
                     acquired or "()", len(gains))
    return out
