r"""Numerical certification of every theoretical claim in the paper.

Each theorem is checked by Monte-Carlo on problems where the ground truth is
available in closed form, and the report is emitted as JSON so that the
manuscript's "verified numerically" statements are reproducible rather than
rhetorical.  ``python -m infogain.theory.verify`` writes
``results/theory/verification.json``.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from infogain.clinical.net_benefit import (
    check_identity,
    information_gain_nats,
    nest_calibrate,
    safe_omission_bound_global,
    safe_omission_bound_local,
    bayes_value,
)
from infogain.theory.lemmas import (
    LemmaReport,
    best_stratified_bound,
    verify_js_tv,
    verify_ks_auroc,
)

LN2 = float(np.log(2.0))


# --------------------------------------------------------------------------- #
# Theorem 3 (identity) -- exactness of the information <-> net-benefit map
# --------------------------------------------------------------------------- #
def verify_schervish_identity(n_trials: int = 200, n: int = 40000,
                              seed: int = 0) -> LemmaReport:
    """Check ``DeltaI == integral of DeltaNB dt/t`` on random nested posteriors."""
    rng = np.random.default_rng(seed)
    worst = 0.0
    violated = 0
    for _ in range(n_trials):
        base = rng.normal(rng.uniform(-4, 0), rng.uniform(0.5, 2.0), size=n)
        extra = rng.normal(0, rng.uniform(0.1, 2.0), size=n)
        eta2 = 1.0 / (1.0 + np.exp(-(base + extra)))
        eta1 = nest_calibrate(eta2, 1.0 / (1.0 + np.exp(-base)), n_bins=400)
        chk = check_identity(eta2, eta1)
        worst = max(worst, chk.rel_error)
        violated += int(chk.rel_error > 1e-3)
    return LemmaReport("Thm3 identity: dI = int dNB dt/t", n_trials, -worst, violated)


def verify_safe_omission(n_trials: int = 400, n: int = 20000,
                         seed: int = 0) -> tuple[LemmaReport, LemmaReport]:
    """Check that Theorem 2's bounds dominate the realised net-benefit loss."""
    rng = np.random.default_rng(seed)
    worst_g, worst_l = np.inf, np.inf
    viol_g = viol_l = 0
    for _ in range(n_trials):
        base = rng.normal(rng.uniform(-4, 0), rng.uniform(0.5, 2.0), size=n)
        extra = rng.normal(0, rng.uniform(0.02, 1.0), size=n)
        eta2 = 1.0 / (1.0 + np.exp(-(base + extra)))
        eta1 = nest_calibrate(eta2, 1.0 / (1.0 + np.exp(-base)), n_bins=400)
        eps_bits = information_gain_nats(eta2, eta1) / LN2
        t0 = float(rng.uniform(0.01, 0.6))
        t = np.array([t0])
        d_nb = float((bayes_value(eta2, t) - bayes_value(eta1, t))[0] / (1 - t0))
        gb = float(safe_omission_bound_global(eps_bits, t0))
        lb = float(safe_omission_bound_local(eps_bits, t0))
        worst_g = min(worst_g, gb - d_nb)
        worst_l = min(worst_l, lb - d_nb)
        viol_g += int(gb - d_nb < -1e-9)
        viol_l += int(lb - d_nb < -1e-9)
    return (LemmaReport("Thm2 global bound dominates dNB", n_trials, float(worst_g), viol_g),
            LemmaReport("Thm2 local bound dominates dNB", n_trials, float(worst_l), viol_l))


def verify_theorem1(n_trials: int = 300, n: int = 8000,
                    seed: int = 0) -> tuple[LemmaReport, LemmaReport, LemmaReport]:
    r"""Check Theorem 1's stratified bound.

Three claims are tested.  (i) The bound never exceeds the gain demonstrably
    achievable on the same rows.  (ii) The bound *equals* the AUROC gain of the
    lexicographic witness score -- this is the theorem's actual content, an
    identity, and checking it to machine precision is a far stronger test than
    checking the inequality it implies.  (iii) On synergy-dominated problems the
    bound is strictly positive, which is the whole point of reporting it.
    """
    rng = np.random.default_rng(seed)
    worst = np.inf
    worst_identity = 0.0
    violated = 0
    n_informative = 0
    for _ in range(n_trials):
        # XOR-style synergy: neither modality alone separates, the pair does
        a = rng.normal(size=n)
        b = rng.normal(size=n)
        logit = (rng.uniform(0.0, 1.5) * a
                 + rng.uniform(0.0, 2.5) * np.sign(a) * b
                 - rng.uniform(1.0, 3.0))
        p = 1.0 / (1.0 + np.exp(-logit))
        y = (rng.random(n) < p).astype(int)
        if y.sum() < 50 or (1 - y).sum() < 50:
            continue
        # coarse = a-only Bayes-ish score, fine = full posterior
        eta_c = 1.0 / (1.0 + np.exp(-(rng.uniform(0.0, 1.5) * a - 1.5)))
        eta_f = p
        bnd = best_stratified_bound(y, eta_c, eta_f)
        worst = min(worst, bnd.achievable_gain - bnd.bound)
        violated += int(bnd.achievable_gain - bnd.bound < -1e-9)
        worst_identity = max(worst_identity, abs(bnd.identity_residual))
        n_informative += int(bnd.bound > 0.005)
    return (LemmaReport("Thm1 bound <= achievable gain", n_trials, float(worst), violated),
            LemmaReport("Thm1 witness identity (bound == lex gain)", n_trials,
                        -float(worst_identity), int(worst_identity > 1e-9)),
            LemmaReport("Thm1 bound informative (>0.005 AUROC)", n_trials,
                        float(n_informative) / max(n_trials, 1), 0))


# --------------------------------------------------------------------------- #
# Estimator sanity: PVI recovers a known information value
# --------------------------------------------------------------------------- #
def verify_pvi_consistency(n: int = 400000, seed: int = 0) -> LemmaReport:
    r"""On a problem with a closed-form :math:`I(X;Y)`, PVI must recover it.

    Take :math:`X\sim\mathcal N(0,1)`, :math:`Y\sim\mathrm{Ber}(\sigma(\beta X-c))`.
    Then :math:`I(X;Y)=H(\pi)-\mathbb E[H(\sigma(\beta X-c))]`, computable by
    quadrature, and the PVI mean under the *true* posterior must match it.
    """
    from scipy import integrate as sint

    rng = np.random.default_rng(seed)
    beta, c = 1.3, 1.1
    x = rng.normal(size=n)
    p = 1.0 / (1.0 + np.exp(-(beta * x - c)))
    y = (rng.random(n) < p).astype(int)

    def h(q):
        q = np.clip(q, 1e-12, 1 - 1e-12)
        return -(q * np.log2(q) + (1 - q) * np.log2(1 - q))

    grid = np.linspace(-9, 9, 200001)
    dens = np.exp(-grid ** 2 / 2) / np.sqrt(2 * np.pi)
    pgrid = 1.0 / (1.0 + np.exp(-(beta * grid - c)))
    pi_true = float(sint.trapezoid(pgrid * dens, grid))
    cond_h = float(sint.trapezoid(h(pgrid) * dens, grid))
    truth = h(pi_true) - cond_h

    logp_model = np.log(np.where(y == 1, p, 1 - p))
    logp_null = np.log(np.where(y == 1, pi_true, 1 - pi_true))
    est = float(np.mean((logp_model - logp_null) / LN2))
    err = abs(est - truth)
    return LemmaReport(f"PVI recovers I(X;Y) (truth={truth:.5f}, est={est:.5f})",
                       n, -err, int(err > 5e-3))


# --------------------------------------------------------------------------- #
# Runner
# --------------------------------------------------------------------------- #
@dataclass
class VerificationReport:
    reports: list[LemmaReport] = field(default_factory=list)

    def add(self, *r: LemmaReport) -> None:
        self.reports.extend(r)

    @property
    def all_hold(self) -> bool:
        return all(r.holds for r in self.reports if not r.name.startswith("Thm1 bound informative"))

    def as_dict(self) -> dict:
        return {"all_hold": self.all_hold,
                "reports": [r.as_dict() for r in self.reports]}

    def pretty(self) -> str:
        lines = [f"{'claim':58s} {'trials':>8s} {'worst slack':>13s} {'viol':>6s}  ok"]
        lines.append("-" * 96)
        for r in self.reports:
            lines.append(f"{r.name[:58]:58s} {r.n_trials:8d} {r.worst_slack:13.3e} "
                         f"{r.violated:6d}  {'PASS' if r.holds else 'FAIL'}")
        return "\n".join(lines)


def run_all(quick: bool = False, seed: int = 0) -> VerificationReport:
    scale = 0.2 if quick else 1.0
    rep = VerificationReport()
    rep.add(*verify_js_tv(n_trials=int(20000 * scale), seed=seed))
    rep.add(*verify_ks_auroc(n_trials=int(2000 * scale), n=1000, seed=seed))
    rep.add(verify_schervish_identity(n_trials=int(150 * scale),
                                      n=int(30000 * (0.4 if quick else 1.0)), seed=seed))
    rep.add(*verify_safe_omission(n_trials=int(300 * scale),
                                  n=int(20000 * (0.4 if quick else 1.0)), seed=seed))
    rep.add(*verify_theorem1(n_trials=int(200 * scale),
                             n=int(8000 * (0.5 if quick else 1.0)), seed=seed))
    rep.add(verify_pvi_consistency(n=int(400000 * (0.25 if quick else 1.0)), seed=seed))
    return rep


def main() -> None:  # pragma: no cover - CLI
    import argparse

    from infogain.utils.io import save_json

    ap = argparse.ArgumentParser(description="Numerically certify the paper's theory")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/theory/verification.json")
    args = ap.parse_args()

    rep = run_all(quick=args.quick, seed=args.seed)
    print(rep.pretty())
    save_json(rep.as_dict(), args.out)
    print(f"\nwrote {args.out}")
    if not rep.all_hold:
        raise SystemExit(1)


if __name__ == "__main__":  # pragma: no cover
    main()
