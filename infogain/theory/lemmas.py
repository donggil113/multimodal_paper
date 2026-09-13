r"""Lemmas linking usable information to discrimination (AUROC).

AUROC is *not* a proper scoring rule, so no unconditional map from bits to
AUROC gain exists — a model can gain information purely by recalibrating and
leave every ranking untouched.  We therefore prove what is actually true and
say so plainly in the paper:

**L1 (Neyman-Pearson).**  Among all scores measurable w.r.t. :math:`X_T`, the
posterior :math:`\eta_T` maximises AUROC.  Write :math:`A^*_T = A(\eta_T)`.

**L2 (KS sandwich).**  Let :math:`D^+=\sup_t[F_0(t)-F_1(t)]` be the one-sided
Kolmogorov-Smirnov statistic, i.e. Youden's J at its best threshold.  Then for
any score :math:`2A-1 \le 2D^+`, and for any score whose ROC is concave (in
particular the posterior, and in general the ROC convex hull, which is
achievable by monotone recalibration) :math:`2A-1 \ge D^+`.  Concavity is not
decoration: a score whose empirical ROC dips below its own chord can have
:math:`2A-1 < D^+`, so the lower bound is stated for
:func:`auroc_hull`.  For the posterior score
:math:`D^+=\mathrm{TV}(P_{X|1},P_{X|0})`, because thresholding the posterior is
thresholding the likelihood ratio and total variation is attained there.

**L3 (information vs. TV).**  With :math:`Y` binary of prevalence :math:`\pi`,
:math:`I(X;Y)` is the generalised Jensen-Shannon divergence
:math:`\mathrm{JS}_\pi(P_{X|1},P_{X|0})` and satisfies

.. math:: 8\pi^2(1-\pi)^2\,\mathrm{TV}^2 \;\le\; I \;\le\; H(\pi)\,\mathrm{TV}.

Both directions have short proofs (Supplementary S1) built on the martingale
identity :math:`\mathbb E[\eta]=\pi` and
:math:`\mathrm{TV}=\mathbb E|\eta-\pi| / (2\pi(1-\pi))`: the upper bound from the
two chords of the concave binary entropy, the lower bound from Pinsker plus
Jensen.  A sharper lower constant :math:`2\pi(1-\pi)` (larger by
:math:`1/4\pi(1-\pi)`) holds in every numerical check we have run and is
reported by :func:`verify_js_tv`, but we do not rely on it: it is used nowhere,
and the looser provable constant only widens the upper end of the AUROC bracket,
which stays valid.

**Theorem 1 (stratified, exact).**  For any partition :math:`\{B_k\}` of the
range of :math:`\eta_S` into intervals,

.. math:: A^*_{S\cup m} \;\ge\; A(\eta_S) + \sum_k w_k\big[A_k(\eta_{S\cup m}) - A_k(\eta_S)\big],

with :math:`w_k` the share of case-control pairs inside :math:`B_k`.  The proof
is a one-line ranking argument: the lexicographic score "bin index, ties broken
by :math:`\eta_{S\cup m}`" is :math:`X_{S\cup m}`-measurable, agrees with
:math:`\eta_S` on every across-bin pair, and replaces the within-bin ordering.
In population :math:`A(\eta_S)=A^*_S` by L1, so the sum is a certified lower
bound on the achievable AUROC *gain*; with a fitted head it is a lower bound on
the gain over that head, which is the comparison clinical papers report anyway.
Combining with L2-L3 inside each bin converts the right-hand side into
within-bin usable information, which is exactly the synergy term: bits that only
exist once :math:`X_S` has been conditioned on.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

LN2 = float(np.log(2.0))


# --------------------------------------------------------------------------- #
# Basic discrimination statistics
# --------------------------------------------------------------------------- #
def auroc(y: np.ndarray, s: np.ndarray) -> float:
    """Rank-based AUROC with the mid-rank convention for ties."""
    y = np.asarray(y).astype(int)
    s = np.asarray(s, dtype=np.float64)
    n1 = int(y.sum())
    n0 = int(y.size - n1)
    if n1 == 0 or n0 == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    s_sorted = s[order]
    ranks = np.empty(s.size, dtype=np.float64)
    i = 0
    while i < s.size:
        j = i
        while j + 1 < s.size and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def roc_points(y: np.ndarray, s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Empirical ROC vertices ``(fpr, tpr)``, including (0,0) and (1,1)."""
    y = np.asarray(y).astype(int)
    s = np.asarray(s, dtype=np.float64)
    order = np.argsort(-s, kind="mergesort")
    ys = y[order]
    ss = s[order]
    tp = np.cumsum(ys)
    fp = np.cumsum(1 - ys)
    # keep only the last index of each tied score block
    keep = np.r_[np.diff(ss) != 0, True]
    tpr = np.r_[0.0, tp[keep] / max(int(y.sum()), 1)]
    fpr = np.r_[0.0, fp[keep] / max(int((1 - y).sum()), 1)]
    return fpr, tpr


def auroc_hull(y: np.ndarray, s: np.ndarray) -> float:
    """AUC of the ROC *convex hull* (Provost & Fawcett's ROCCH).

    The hull is achievable by a monotone recalibration of ``s`` (equivalently, by
    randomising between two of its thresholds), so it never exceeds the optimum
    :math:`A^*` attainable from the same inputs, and unlike the raw empirical AUC
    it is concave -- which is what makes the :math:`D^+` lower bound valid.
    """
    fpr, tpr = roc_points(y, s)
    # upper convex hull by monotone chain over points sorted by fpr
    pts = sorted(zip(fpr.tolist(), tpr.tolist()))
    hull: list[tuple[float, float]] = []
    for x, yy in pts:
        while len(hull) >= 2:
            (x1, y1), (x2, y2) = hull[-2], hull[-1]
            # drop hull[-1] if it lies below the chord (x1,y1)-(x,yy)
            if (y2 - y1) * (x - x1) <= (yy - y1) * (x2 - x1):
                hull.pop()
            else:
                break
        hull.append((x, yy))
    hx = np.array([h[0] for h in hull])
    hy = np.array([h[1] for h in hull])
    return float(np.trapezoid(hy, hx)) if hasattr(np, "trapezoid") else float(np.trapz(hy, hx))


def _cdf_gap(y: np.ndarray, s: np.ndarray) -> np.ndarray:
    y = np.asarray(y).astype(int)
    s = np.asarray(s, dtype=np.float64)
    grid = np.unique(s)
    s1 = np.sort(s[y == 1])
    s0 = np.sort(s[y == 0])
    f1 = np.searchsorted(s1, grid, side="right") / s1.size
    f0 = np.searchsorted(s0, grid, side="right") / s0.size
    return f0 - f1


def ks_signed(y: np.ndarray, s: np.ndarray) -> float:
    r"""One-sided KS statistic :math:`D^+ = \sup_t\,[F_0(t) - F_1(t)]`.

    This, not the two-sided :math:`\sup|F_0-F_1|`, is the quantity the AUROC
    sandwich is stated with: :math:`D^+ = \sup_t [\mathrm{TPR}(t)-\mathrm{FPR}(t)]`
    is Youden's J at its best threshold, and it is non-negative by construction
    (the empty rejection region gives 0).  Using the two-sided statistic breaks
    the lower bound for scores that carry no signal, where :math:`2A-1\approx 0`
    while sampling noise keeps :math:`\sup|F_0-F_1|` strictly positive.
    """
    y = np.asarray(y).astype(int)
    if y.sum() == 0 or y.sum() == y.size:
        return float("nan")
    return float(max(0.0, np.max(_cdf_gap(y, s))))


def ks_statistic(y: np.ndarray, s: np.ndarray) -> float:
    """Two-sided :math:`\sup_t |F_0(t) - F_1(t)|` (reported, not used in bounds)."""
    y = np.asarray(y).astype(int)
    if y.sum() == 0 or y.sum() == y.size:
        return float("nan")
    return float(np.max(np.abs(_cdf_gap(y, s))))


def entropy_bits(pi: float) -> float:
    pi = float(np.clip(pi, 1e-12, 1 - 1e-12))
    return float(-(pi * np.log2(pi) + (1 - pi) * np.log2(1 - pi)))


def entropy_nats(pi: float) -> float:
    return entropy_bits(pi) * LN2


# --------------------------------------------------------------------------- #
# L3: generalised Jensen-Shannon vs. total variation
# --------------------------------------------------------------------------- #
def total_variation(p1: np.ndarray, p0: np.ndarray) -> float:
    return float(0.5 * np.abs(np.asarray(p1) - np.asarray(p0)).sum())


def js_pi(p1: np.ndarray, p0: np.ndarray, pi: float, eps: float = 1e-15) -> float:
    r""":math:`\mathrm{JS}_\pi(P_1,P_0) = I(X;Y)` in nats for binary :math:`Y`."""
    p1 = np.clip(np.asarray(p1, dtype=np.float64), eps, None)
    p0 = np.clip(np.asarray(p0, dtype=np.float64), eps, None)
    mix = pi * p1 + (1 - pi) * p0
    return float(pi * np.sum(p1 * np.log(p1 / mix))
                 + (1 - pi) * np.sum(p0 * np.log(p0 / mix)))


@dataclass
class LemmaReport:
    name: str
    n_trials: int
    worst_slack: float
    violated: int

    @property
    def holds(self) -> bool:
        return self.violated == 0

    def as_dict(self) -> dict:
        return {"name": self.name, "n_trials": self.n_trials,
                "worst_slack": self.worst_slack, "violated": self.violated,
                "holds": self.holds}


def verify_js_tv(n_trials: int = 20000, max_support: int = 12, seed: int = 0,
                 tol: float = 1e-10) -> tuple[LemmaReport, LemmaReport, LemmaReport]:
    r"""Monte-Carlo check of the two proved L3 bounds, plus the sharper conjecture.

    Random Dirichlet class-conditionals over supports of size 2..``max_support``,
    plus deterministic edge cases (disjoint supports, identical distributions)
    where the inequalities are tight.
    """
    rng = np.random.default_rng(seed)
    worst_up, worst_lo, worst_sharp = np.inf, np.inf, np.inf
    viol_up = viol_lo = viol_sharp = 0
    for _ in range(n_trials):
        k = int(rng.integers(2, max_support + 1))
        alpha = 10.0 ** rng.uniform(-1.2, 1.0)
        p1 = rng.dirichlet(np.full(k, alpha))
        p0 = rng.dirichlet(np.full(k, alpha))
        pi = float(rng.uniform(0.005, 0.995))
        tv = total_variation(p1, p0)
        info = js_pi(p1, p0, pi)
        up_slack = entropy_nats(pi) * tv - info
        lo_slack = info - 8 * (pi * (1 - pi)) ** 2 * tv ** 2
        sharp_slack = info - 2 * pi * (1 - pi) * tv ** 2
        worst_up = min(worst_up, up_slack)
        worst_lo = min(worst_lo, lo_slack)
        worst_sharp = min(worst_sharp, sharp_slack)
        viol_up += int(up_slack < -tol)
        viol_lo += int(lo_slack < -tol)
        viol_sharp += int(sharp_slack < -tol)
    return (LemmaReport("I <= H(pi) * TV", n_trials, float(worst_up), viol_up),
            LemmaReport("I >= 8 pi^2 (1-pi)^2 TV^2 (proved)", n_trials,
                        float(worst_lo), viol_lo),
            LemmaReport("I >= 2 pi (1-pi) TV^2 (sharper, numerical only)",
                        n_trials, float(worst_sharp), viol_sharp))


def verify_ks_auroc(n_trials: int = 3000, n: int = 400, seed: int = 0,
                    tol: float = 1e-9) -> tuple[LemmaReport, LemmaReport]:
    r"""Monte-Carlo check of :math:`\mathrm{KS} \le 2A-1 \le 2\,\mathrm{KS}`."""
    rng = np.random.default_rng(seed)
    worst_lo, worst_hi = np.inf, np.inf
    viol_lo = viol_hi = 0
    for _ in range(n_trials):
        pi = float(rng.uniform(0.05, 0.95))
        y = (rng.random(n) < pi).astype(int)
        if y.sum() in (0, n):
            continue
        shift = rng.uniform(0, 3)
        scale = rng.uniform(0.3, 3.0)
        s = rng.normal(size=n) * scale + shift * y
        if rng.random() < 0.3:  # heavy ties
            s = np.round(s, 0)
        a, ah, ks = auroc(y, s), auroc_hull(y, s), ks_signed(y, s)
        worst_lo = min(worst_lo, (2 * ah - 1) - ks)
        worst_hi = min(worst_hi, 2 * ks - (2 * a - 1))
        viol_lo += int((2 * ah - 1) - ks < -tol)
        viol_hi += int(2 * ks - (2 * a - 1) < -tol)
    return (LemmaReport("2A_hull-1 >= D+", n_trials, float(worst_lo), viol_lo),
            LemmaReport("2A-1 <= 2 D+", n_trials, float(worst_hi), viol_hi))


# --------------------------------------------------------------------------- #
# Information -> achievable AUROC (level bounds)
# --------------------------------------------------------------------------- #
def achievable_auroc_bracket(info_bits: float, prevalence: float,
                             sharp: bool = False) -> tuple[float, float]:
    r"""Bracket :math:`A^*` from :math:`I_{\mathcal V}` alone via L2+L3.

    Lower: :math:`A^* \ge \tfrac12 + \tfrac12\,I/H(\pi)`, from
    :math:`\mathrm{TV}\ge I/H(\pi)` and the concave-ROC side of L2.
    Upper: :math:`A^* \le \tfrac12 + \sqrt{I}/(2\sqrt2\,\pi(1-\pi))`, from the
    *proved* L3 lower bound.  ``sharp=True`` substitutes the tighter constant
    that is numerically verified but not proved; it is off by default because
    only the loose version is a theorem, and the loose version is still a valid
    upper bound.
    """
    info = max(float(info_bits), 0.0) * LN2
    pi = float(np.clip(prevalence, 1e-6, 1 - 1e-6))
    lo = 0.5 + 0.5 * min(1.0, info / entropy_nats(pi))
    denom = (2 * pi * (1 - pi)) if sharp else (8 * (pi * (1 - pi)) ** 2)
    hi = 0.5 + min(0.5, float(np.sqrt(info / denom)))
    return lo, max(hi, lo)


# --------------------------------------------------------------------------- #
# Theorem 1: stratified achievable-AUROC gain
# --------------------------------------------------------------------------- #
@dataclass
class StratifiedAurocBound:
    bound: float
    n_bins: int
    observed_gain: float
    auroc_coarse: float
    auroc_fine: float
    per_bin: list[dict]

    def as_dict(self) -> dict:
        return {"bound": self.bound, "n_bins": self.n_bins,
                "observed_gain": self.observed_gain,
                "auroc_coarse": self.auroc_coarse, "auroc_fine": self.auroc_fine}


def stratified_auroc_gain_bound(y: np.ndarray, eta_coarse: np.ndarray,
                                eta_fine: np.ndarray, n_bins: int = 10,
                                min_bin: int = 30) -> StratifiedAurocBound:
    r"""Theorem 1's certified lower bound on the achievable AUROC gain.

    Bins are equal-frequency in :math:`\eta_S`.  ``w_k`` is the exact share of
    (case, control) pairs falling inside bin :math:`k`, so the returned bound is
    a valid lower bound on :math:`A^*_{S\cup m} - A^*_S` for the *particular*
    partition used; the paper maximises over ``n_bins``.
    """
    y = np.asarray(y).astype(int)
    ec = np.asarray(eta_coarse, dtype=np.float64)
    ef = np.asarray(eta_fine, dtype=np.float64)
    n1, n0 = int(y.sum()), int((1 - y).sum())
    if n1 == 0 or n0 == 0:
        raise ValueError("need both classes")

    edges = np.quantile(ec, np.linspace(0, 1, n_bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    bin_id = np.clip(np.searchsorted(edges, ec, side="right") - 1, 0, n_bins - 1)

    total, rows = 0.0, []
    for k in range(n_bins):
        sel = bin_id == k
        if sel.sum() < min_bin:
            continue
        yk = y[sel]
        k1, k0 = int(yk.sum()), int((1 - yk).sum())
        if k1 == 0 or k0 == 0:
            continue
        w_k = (k1 * k0) / (n1 * n0)
        a_fine = auroc(yk, ef[sel])
        a_coarse = auroc(yk, ec[sel])
        contrib = w_k * (a_fine - a_coarse)
        total += contrib
        rows.append({"bin": k, "n": int(sel.sum()), "prevalence": k1 / sel.sum(),
                     "pair_weight": w_k, "auroc_fine": a_fine,
                     "auroc_coarse": a_coarse, "contribution": contrib})

    return StratifiedAurocBound(
        bound=float(total), n_bins=n_bins,
        observed_gain=float(auroc(y, ef) - auroc(y, ec)),
        auroc_coarse=float(auroc(y, ec)), auroc_fine=float(auroc(y, ef)),
        per_bin=rows)


def best_stratified_bound(y: np.ndarray, eta_coarse: np.ndarray,
                          eta_fine: np.ndarray,
                          bin_grid: tuple[int, ...] = (1, 2, 4, 6, 8, 10, 15, 20, 30),
                          min_bin: int = 30) -> StratifiedAurocBound:
    """Maximise Theorem 1's bound over equal-frequency partitions."""
    best: StratifiedAurocBound | None = None
    for nb in bin_grid:
        if len(y) // max(nb, 1) < min_bin:
            continue
        try:
            cand = stratified_auroc_gain_bound(y, eta_coarse, eta_fine, nb, min_bin)
        except ValueError:
            continue
        if best is None or cand.bound > best.bound:
            best = cand
    if best is None:
        raise ValueError("no admissible partition; cohort too small")
    return best


def within_bin_information_bound(y: np.ndarray, eta_coarse: np.ndarray,
                                 delta_bits: np.ndarray, n_bins: int = 10,
                                 min_bin: int = 30) -> float:
    r"""Information-only form of Theorem 1.

    Replaces the within-bin AUROC of the fine model with the L2+L3 lower bound
    :math:`\tfrac12 + \tfrac12 U_k / H(\pi_k)` driven by the within-bin
    conditional usable information ``delta_bits``.  This is the version quoted in
    the paper, because its right-hand side is a *decomposition term* rather than
    a second model's performance.
    """
    y = np.asarray(y).astype(int)
    ec = np.asarray(eta_coarse, dtype=np.float64)
    db = np.asarray(delta_bits, dtype=np.float64)
    n1, n0 = int(y.sum()), int((1 - y).sum())
    edges = np.quantile(ec, np.linspace(0, 1, n_bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    bin_id = np.clip(np.searchsorted(edges, ec, side="right") - 1, 0, n_bins - 1)

    total = 0.0
    for k in range(n_bins):
        sel = bin_id == k
        if sel.sum() < min_bin:
            continue
        yk = y[sel]
        k1, k0 = int(yk.sum()), int((1 - yk).sum())
        if k1 == 0 or k0 == 0:
            continue
        w_k = (k1 * k0) / (n1 * n0)
        pi_k = k1 / sel.sum()
        u_k = max(float(np.mean(db[sel])), 0.0) * LN2
        gain_k = 0.5 * min(1.0, u_k / entropy_nats(pi_k))
        penalty_k = auroc(yk, ec[sel]) - 0.5
        total += w_k * max(gain_k - penalty_k, 0.0)
    return float(total)
