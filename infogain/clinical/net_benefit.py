r"""Decision-curve analysis and the exact information :math:`\leftrightarrow` net-benefit map.

The paper's central translation result is an *identity*, not a bound.  Write
:math:`\eta_1 = P(Y=1\mid X_S)`, :math:`\eta_2 = P(Y=1\mid X_{S\cup m})`, and let

.. math:: V_j(t) = \mathbb{E}\big[(\eta_j - t)_+\big], \qquad
          \mathrm{NB}_j(t) = \frac{V_j(t)}{1-t}

be the Bayes value and Vickers' net benefit at risk threshold :math:`t` (i.e. the
threshold at which a clinician is indifferent between treating and not treating,
so the harm of one unnecessary treatment equals :math:`t/(1-t)` times the harm of
one missed case).  If :math:`\eta_1 = \mathbb{E}[\eta_2\mid X_S]` — the nesting
condition, enforced in practice by recalibrating the coarse model onto the fine
one — then

.. math::
    \Delta I \;=\; \mathbb{E}\,\mathrm{KL}\!\big(\mathrm{Ber}(\eta_2)\,\|\,\mathrm{Ber}(\eta_1)\big)
    \;=\; \int_0^1 \big[V_2(t)-V_1(t)\big]\,\frac{\mathrm{d}t}{t(1-t)}
    \;=\; \int_0^1 \Delta \mathrm{NB}(t)\,\frac{\mathrm{d}t}{t}\,,

in nats.  Information gain is *literally* a weighted integral of net-benefit
gains over every threshold a clinician might hold.  Two consequences drive the
rest of the paper:

* restricting the integral to a clinically defensible threshold window
  :math:`[a,b]` defines **clinically restricted usable information**, which is
  what should be reported instead of the unrestricted bit count; and
* :math:`\Delta I \le \varepsilon` forces :math:`\Delta\mathrm{NB}(t)` to be
  small at *every* threshold, which is the safety certificate behind omitting a
  test (Theorem 2).

Reference for the threshold weighting: Schervish (1989); for net benefit:
Vickers & Elkin (2006).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import integrate, optimize

LN2 = float(np.log(2.0))


# --------------------------------------------------------------------------- #
# Net benefit
# --------------------------------------------------------------------------- #
def empirical_net_benefit(y: np.ndarray, p: np.ndarray,
                          thresholds: np.ndarray) -> np.ndarray:
    """Vickers net benefit computed from realised labels.

    :math:`\\mathrm{NB}(t) = \\frac{TP}{n} - \\frac{FP}{n}\\cdot\\frac{t}{1-t}`
    where the rule treats iff :math:`p_i > t`.
    """
    y = np.asarray(y).astype(np.float64)
    p = np.asarray(p, dtype=np.float64)
    t = np.asarray(thresholds, dtype=np.float64)
    n = y.size
    order = np.argsort(p, kind="mergesort")
    ps, ys = p[order], y[order]
    # suffix sums over patients with p > t, found by binary search per threshold
    cum_y = np.concatenate([[0.0], np.cumsum(ys)])
    cum_1 = np.arange(n + 1, dtype=np.float64)
    idx = np.searchsorted(ps, t, side="right")
    tp = (cum_y[n] - cum_y[idx]) / n
    n_treat = (cum_1[n] - cum_1[idx]) / n
    fp = n_treat - tp
    w = t / np.clip(1.0 - t, 1e-12, None)
    return tp - fp * w


def model_net_benefit(p: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    """Net benefit *implied by the model's own risks*, ``E[(p-t)+]/(1-t)``.

    This is the population quantity in the identity above.  Comparing it against
    :func:`empirical_net_benefit` is a calibration check: the two agree exactly
    when ``p`` is calibrated, and their gap is reported alongside every decision
    curve in the paper.
    """
    t = np.asarray(thresholds, dtype=np.float64)
    return bayes_value(p, t) / np.clip(1.0 - t, 1e-12, None)


def bayes_value(p: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    """:math:`V(t) = \\mathbb E[(p-t)_+]`, the unnormalised Bayes value."""
    p = np.asarray(p, dtype=np.float64)
    t = np.asarray(thresholds, dtype=np.float64)
    n = p.size
    ps = np.sort(p)
    cum = np.concatenate([[0.0], np.cumsum(ps)])
    idx = np.searchsorted(ps, t, side="right")
    tail_sum = cum[n] - cum[idx]
    tail_cnt = (n - idx).astype(np.float64)
    return (tail_sum - t * tail_cnt) / n


def treat_all_net_benefit(prevalence: float, thresholds: np.ndarray) -> np.ndarray:
    t = np.asarray(thresholds, dtype=np.float64)
    return prevalence - (1.0 - prevalence) * t / np.clip(1.0 - t, 1e-12, None)


@dataclass
class DecisionCurve:
    thresholds: np.ndarray
    net_benefit: np.ndarray
    treat_all: np.ndarray
    treat_none: np.ndarray
    label: str = ""

    def standardized(self, prevalence: float) -> np.ndarray:
        """Net benefit divided by prevalence: 'true positives per 100 patients'."""
        return self.net_benefit / max(prevalence, 1e-12)


def decision_curve(y: np.ndarray, p: np.ndarray, thresholds: np.ndarray,
                   label: str = "") -> DecisionCurve:
    prev = float(np.mean(y))
    return DecisionCurve(
        thresholds=np.asarray(thresholds, dtype=np.float64),
        net_benefit=empirical_net_benefit(y, p, thresholds),
        treat_all=treat_all_net_benefit(prev, thresholds),
        treat_none=np.zeros_like(np.asarray(thresholds, dtype=np.float64)),
        label=label,
    )


# --------------------------------------------------------------------------- #
# The information <-> net-benefit identity
# --------------------------------------------------------------------------- #
def kl_bernoulli(p: np.ndarray, q: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=np.float64), eps, 1 - eps)
    q = np.clip(np.asarray(q, dtype=np.float64), eps, 1 - eps)
    return p * np.log(p / q) + (1 - p) * np.log((1 - p) / (1 - q))


def information_gain_nats(eta_fine: np.ndarray, eta_coarse: np.ndarray) -> float:
    r""":math:`\Delta I = \mathbb E\,\mathrm{KL}(\mathrm{Ber}(\eta_2)\|\mathrm{Ber}(\eta_1))` in nats."""
    return float(np.mean(kl_bernoulli(eta_fine, eta_coarse)))


def threshold_integral_nats(eta_fine: np.ndarray, eta_coarse: np.ndarray,
                            t_lo: float = 1e-4, t_hi: float = 1 - 1e-4,
                            n_grid: int = 20001, log_grid: bool = True) -> float:
    r"""Numerically evaluate :math:`\int \Delta\mathrm{NB}(t)\,\mathrm{d}t/t`.

    A log-spaced grid is used because the integrand's mass concentrates near the
    small thresholds that matter clinically (treat-if-risk-above-2 % problems).
    """
    if log_grid:
        t = np.exp(np.linspace(np.log(t_lo), np.log(t_hi), n_grid))
    else:
        t = np.linspace(t_lo, t_hi, n_grid)
    dV = bayes_value(eta_fine, t) - bayes_value(eta_coarse, t)
    integrand = dV / (t * (1.0 - t))
    return float(integrate.trapezoid(integrand, t))


def restricted_information(eta_fine: np.ndarray, eta_coarse: np.ndarray,
                           t_lo: float, t_hi: float, n_grid: int = 4001,
                           in_bits: bool = True) -> float:
    r"""**Clinically restricted usable information** :math:`I^{[a,b]}`.

    The share of the information gain that is redeemable at risk thresholds a
    clinician would actually hold.  Bits earned outside :math:`[a,b]` buy nothing
    and should not be counted when a test is being justified.
    """
    val = threshold_integral_nats(eta_fine, eta_coarse, t_lo, t_hi, n_grid)
    return val / LN2 if in_bits else val


@dataclass
class IdentityCheck:
    kl_nats: float
    integral_nats: float
    abs_error: float
    rel_error: float

    @property
    def passes(self) -> bool:
        return self.rel_error < 1e-2


def check_identity(eta_fine: np.ndarray, eta_coarse: np.ndarray,
                   **kw) -> IdentityCheck:
    """Verify the Schervish identity numerically on a given pair of risk vectors.

    Requires the nesting condition ``eta_coarse = E[eta_fine | X_S]``; use
    :func:`nest_calibrate` first when the two models were fitted independently.
    """
    kl = information_gain_nats(eta_fine, eta_coarse)
    integ = threshold_integral_nats(eta_fine, eta_coarse, **kw)
    err = abs(kl - integ)
    return IdentityCheck(kl, integ, err, err / max(abs(kl), 1e-12))


def nest_calibrate(eta_fine: np.ndarray, eta_coarse: np.ndarray,
                   n_bins: int | None = None) -> np.ndarray:
    r"""Project the coarse risks onto :math:`\mathbb E[\eta_2 \mid \eta_1]`.

    Two separately-trained heads need not satisfy the nesting condition that
    Theorem 3 is stated under, so the coarse model is replaced by the conditional
    mean of the fine one given the coarse ranking.

    Isotonic regression of :math:`\eta_2` on :math:`\eta_1`, rather than binning.
    Fixed bins smooth within each bin whether or not there is anything to smooth,
    which shows up as a spurious information gap: with
    :math:`\eta_2=\eta_1` the projection should be the identity and the gap
    exactly zero, and only the isotonic fit gives that (a monotone sequence is
    its own isotonic fit).  Passing ``n_bins`` forces the old binned estimator,
    which is kept for the degenerate case of a constant coarse model.
    """
    fine = np.asarray(eta_fine, dtype=np.float64)
    coarse = np.asarray(eta_coarse, dtype=np.float64)
    if n_bins is None and coarse.size > 2 and np.ptp(coarse) > 0:
        from sklearn.isotonic import IsotonicRegression

        iso = IsotonicRegression(y_min=0.0, y_max=1.0, increasing=True,
                                 out_of_bounds="clip")
        return np.clip(iso.fit_transform(coarse, fine), 1e-9, 1 - 1e-9)

    order = np.argsort(coarse)
    out = np.empty_like(coarse)
    chunks = np.array_split(order, min(n_bins or 50, max(1, coarse.size // 20)))
    for idx in chunks:
        if idx.size:
            out[idx] = fine[idx].mean()
    return out


# --------------------------------------------------------------------------- #
# Theorem 2: safe omission
# --------------------------------------------------------------------------- #
def safe_omission_bound_global(eps_bits: float, t: float | np.ndarray
                               ) -> float | np.ndarray:
    r"""Global bound: :math:`\Delta\mathrm{NB}(t) \le \sqrt{\varepsilon/2}\,/\,(1-t)`.

    Derivation: :math:`\Delta V \ge 0`, :math:`|\Delta V'|\le 1` and
    :math:`\nu(t)=1/(t(1-t)) \ge 4` give :math:`\int \Delta V \le \varepsilon/4`;
    a non-negative 1-Lipschitz function with integral :math:`A` has supremum at
    most :math:`\sqrt{2A}`.  ``eps_bits`` is the information gain the omitted
    test would have provided.
    """
    eps = max(float(eps_bits), 0.0) * LN2
    return float(np.sqrt(eps / 2.0)) / np.clip(1.0 - np.asarray(t, dtype=np.float64), 1e-12, None)


def safe_omission_bound_local(eps_bits: float, t0: float) -> float:
    r"""Threshold-localised (much tighter) version of Theorem 2.

    Instead of using only :math:`\nu \ge 4`, integrate the true weight
    :math:`\nu(t)=1/(t(1-t))` against the worst-case tent
    :math:`\Delta V(t) \ge (v - |t-t_0|)_+`.  Solving

    .. math:: \varepsilon \;=\; \int_{t_0-v}^{t_0+v} (v-|t-t_0|)\,\nu(t)\,\mathrm dt

    for :math:`v` gives the largest Bayes-value loss compatible with an
    information gain of :math:`\varepsilon`; the net-benefit loss is
    :math:`v/(1-t_0)`.  At the small thresholds used in triage this is several
    times tighter than the global bound because :math:`\nu` blows up there.
    """
    eps = max(float(eps_bits), 0.0) * LN2
    if eps <= 0:
        return 0.0
    t0 = float(np.clip(t0, 1e-6, 1 - 1e-6))

    def mass(v: float) -> float:
        lo, hi = max(1e-9, t0 - v), min(1 - 1e-9, t0 + v)
        if hi <= lo:
            return 0.0
        tt = np.linspace(lo, hi, 2001)
        tent = np.maximum(v - np.abs(tt - t0), 0.0)
        return float(integrate.trapezoid(tent / (tt * (1.0 - tt)), tt))

    hi_v = 1.0
    if mass(hi_v) < eps:
        return safe_omission_bound_global(eps_bits, t0)
    v = optimize.brentq(lambda vv: mass(vv) - eps, 1e-12, hi_v, xtol=1e-10)
    return float(v) / (1.0 - t0)


def cohort_omission_loss(delta_bits: np.ndarray, t: float,
                         localized: bool = True) -> float:
    r"""Certified net-benefit loss for omitting a test in a whole cohort.

    The per-patient bound is concave in :math:`\varepsilon`, so by Jensen the
    cohort-average loss is bounded by the bound evaluated at the *average*
    forgone information.  Patients for whom the test is kept contribute zero.
    """
    d = np.asarray(delta_bits, dtype=np.float64)
    d = d[np.isfinite(d)]
    if d.size == 0:
        return 0.0
    mean_bits = float(np.mean(np.maximum(d, 0.0)))
    return (safe_omission_bound_local(mean_bits, t) if localized
            else float(safe_omission_bound_global(mean_bits, t)))


def min_detectable_gain(t: float, nb_resolution: float) -> float:
    """Inverse of Theorem 2: bits needed before net benefit can move by ``nb_resolution``.

    Useful for powering a study: a test whose information gain is below this
    value provably cannot shift the decision curve by a clinically visible
    amount at threshold ``t``.
    """
    lo, hi = 1e-9, 10.0
    f = lambda b: safe_omission_bound_local(b, t) - nb_resolution  # noqa: E731
    if f(hi) < 0:
        return hi
    if f(lo) > 0:
        return lo
    return float(optimize.brentq(f, lo, hi, xtol=1e-12))
