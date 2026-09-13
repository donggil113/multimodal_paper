r"""Finite-sample bounds for :math:`\mathcal V`-information estimates.

The estimator :math:`\hat I_{\mathcal V}` is a held-out sample mean of a bounded
random variable (PVI, clipped to :math:`\pm C` bits), evaluated at a *particular*
trained predictor :math:`\hat f_S` rather than at the family infimum.  Those two
facts give an asymmetric guarantee, which we keep asymmetric rather than papering
over (Theorem 3 in the paper):

* **Lower bound (always valid).**  Any :math:`f\in\mathcal V` witnesses
  :math:`H_{\mathcal V}(Y|X_S)\le \mathbb E[-\log f]`, so with probability
  :math:`1-\delta`

  .. math:: I_{\mathcal V}(X_S\to Y) \;\ge\; \hat I_{\mathcal V} - \varepsilon_{\mathrm{EB}}(n,\delta).

* **Upper bound (needs capacity control).**  Closing the gap to the infimum costs
  a uniform-convergence term, here the Rademacher complexity of the head class,

  .. math:: I_{\mathcal V}(X_S\to Y) \;\le\; \hat I_{\mathcal V}
            + \varepsilon_{\mathrm{EB}}(n,\delta) + 2 L_\ell \mathfrak R_n(\mathcal V) + \gamma_{\mathrm{opt}}.

Everything is reported in bits.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

_Z975 = 1.959963984540054


# --------------------------------------------------------------------------- #
# Concentration
# --------------------------------------------------------------------------- #
def empirical_bernstein_ci(x: np.ndarray, alpha: float = 0.05,
                           rng_width: float | None = None) -> tuple[float, float]:
    """Two-sided Maurer & Pontil (2009) empirical-Bernstein interval.

    For i.i.d. ``x`` in an interval of width ``R``, with probability
    :math:`\\ge 1-\\alpha`,

    .. math:: |\\bar x - \\mathbb E x| \\le \\sqrt{\\frac{2\\hat V\\ln(4/\\alpha)}{n}}
              + \\frac{7R\\ln(4/\\alpha)}{3(n-1)}.

    Unlike Hoeffding this adapts to the *observed* PVI variance, which is small
    for well-calibrated heads and large exactly where the estimate should not be
    trusted.
    """
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    if n < 2:
        m = float(x.mean()) if n else float("nan")
        return m, m
    mean = float(x.mean())
    var = float(x.var(ddof=1))
    R = float(rng_width) if rng_width is not None else float(x.max() - x.min())
    log_term = math.log(4.0 / alpha)
    eps = math.sqrt(2.0 * var * log_term / n) + 7.0 * R * log_term / (3.0 * (n - 1))
    return mean - eps, mean + eps


def hoeffding_ci(x: np.ndarray, alpha: float = 0.05,
                 rng_width: float | None = None) -> tuple[float, float]:
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    if n < 1:
        return float("nan"), float("nan")
    R = float(rng_width) if rng_width is not None else float(x.max() - x.min())
    eps = R * math.sqrt(math.log(2.0 / alpha) / (2.0 * n))
    return float(x.mean()) - eps, float(x.mean()) + eps


def bootstrap_ci(x: np.ndarray, alpha: float = 0.05, n_boot: int = 4000,
                 seed: int = 0, kind: str = "bca") -> tuple[float, float]:
    """Percentile / basic / BCa bootstrap interval for the mean."""
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    if n < 2:
        m = float(x.mean()) if n else float("nan")
        return m, m
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    boots = x[idx].mean(axis=1)
    theta = float(x.mean())

    if kind == "percentile":
        return tuple(np.quantile(boots, [alpha / 2, 1 - alpha / 2]))  # type: ignore[return-value]
    if kind == "basic":
        lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
        return 2 * theta - hi, 2 * theta - lo
    if kind != "bca":
        raise ValueError(kind)

    # BCa: bias correction z0 + acceleration a from the jackknife.
    from scipy.stats import norm

    prop = float(np.mean(boots < theta))
    prop = min(max(prop, 1.0 / n_boot), 1.0 - 1.0 / n_boot)
    z0 = norm.ppf(prop)
    total = x.sum()
    jack = (total - x) / (n - 1)
    jbar = jack.mean()
    num = float(((jbar - jack) ** 3).sum())
    den = float(6.0 * (((jbar - jack) ** 2).sum() ** 1.5))
    a = num / den if den > 0 else 0.0

    def _adj(q: float) -> float:
        z = norm.ppf(q)
        return float(norm.cdf(z0 + (z0 + z) / max(1e-12, 1 - a * (z0 + z))))

    lo_q, hi_q = _adj(alpha / 2), _adj(1 - alpha / 2)
    lo_q = min(max(lo_q, 1e-6), 1 - 1e-6)
    hi_q = min(max(hi_q, 1e-6), 1 - 1e-6)
    return tuple(np.quantile(boots, [lo_q, hi_q]))  # type: ignore[return-value]


def paired_permutation_pvalue(a: np.ndarray, b: np.ndarray, n_perm: int = 10000,
                              seed: int = 0, alternative: str = "greater") -> float:
    """Sign-flip permutation test for ``mean(a - b) > 0`` (paired PVI gains)."""
    d = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    obs = float(d.mean())
    rng = np.random.default_rng(seed)
    signs = rng.choice([-1.0, 1.0], size=(n_perm, d.size))
    null = (signs * d).mean(axis=1)
    if alternative == "greater":
        hits = int((null >= obs).sum())
    elif alternative == "less":
        hits = int((null <= obs).sum())
    else:
        hits = int((np.abs(null) >= abs(obs)).sum())
    return (hits + 1) / (n_perm + 1)


def dkw_band(n: int, alpha: float = 0.05) -> float:
    """Dvoretzky-Kiefer-Wolfowitz uniform band half-width for an empirical CDF.

    Used to turn the pointwise decision-curve estimate into a band that is valid
    simultaneously over all risk thresholds, which is what a "reduce X% of tests
    at no loss" claim actually needs.
    """
    return math.sqrt(math.log(2.0 / alpha) / (2.0 * n))


# --------------------------------------------------------------------------- #
# Capacity of the predictive family
# --------------------------------------------------------------------------- #
@dataclass
class CapacityBound:
    """Rademacher-complexity bound for the fusion head, in nats of log-loss."""

    rademacher: float
    depth: int
    frob_product: float
    input_radius: float
    n: int

    @property
    def loss_penalty_bits(self) -> float:
        """``2 * L_loss * R_n`` converted to bits.

        The log-loss is 1-Lipschitz in the logit on the region where predicted
        probabilities stay inside ``[p_min, 1-p_min]``; we use ``L=1`` and note
        the caveat in the paper rather than inflating the constant.
        """
        return 2.0 * self.rademacher / math.log(2.0)


def mlp_rademacher(weight_frob_norms: list[float], input_radius: float,
                   n: int) -> CapacityBound:
    r"""Golowich, Rakhlin & Shamir (2018) bound for depth-:math:`L` ReLU nets.

    .. math:: \mathfrak R_n(\mathcal V) \le
              \frac{B\left(\prod_{l=1}^{L}\|W_l\|_F\right)\sqrt{2L\log 2}}{\sqrt n}

    This is loose in absolute terms (as all norm-based bounds are), but it is
    *computable from the trained weights*, it is monotone in exactly the
    quantities we control by weight decay, and it makes the upper-bound side of
    Theorem 3 a checkable statement rather than an asymptotic gesture.
    """
    L = len(weight_frob_norms)
    prod = float(np.prod(weight_frob_norms)) if L else 0.0
    rad = input_radius * prod * math.sqrt(2.0 * max(L, 1) * math.log(2.0)) / math.sqrt(max(n, 1))
    return CapacityBound(rademacher=rad, depth=L, frob_product=prod,
                         input_radius=float(input_radius), n=int(n))


@dataclass
class TwoSidedVInfoBound:
    """The asymmetric guarantee of Theorem 3, materialised."""

    estimate: float
    lower: float
    upper: float
    deviation: float
    capacity_penalty: float
    optimization_gap: float
    alpha: float

    def as_dict(self) -> dict:
        return {
            "estimate": self.estimate, "lower": self.lower, "upper": self.upper,
            "deviation": self.deviation, "capacity_penalty": self.capacity_penalty,
            "optimization_gap": self.optimization_gap, "alpha": self.alpha,
        }


def two_sided_bound(pvi: np.ndarray, capacity: CapacityBound | None = None,
                    optimization_gap: float = 0.0, alpha: float = 0.05,
                    clip: float = 8.0) -> TwoSidedVInfoBound:
    """Assemble the Theorem-3 bracket around a PVI sample mean."""
    lo, hi = empirical_bernstein_ci(pvi, alpha=alpha, rng_width=2 * clip)
    est = float(np.mean(pvi))
    dev = est - lo
    pen = capacity.loss_penalty_bits if capacity is not None else 0.0
    return TwoSidedVInfoBound(estimate=est, lower=lo, upper=hi + pen + optimization_gap,
                              deviation=dev, capacity_penalty=pen,
                              optimization_gap=optimization_gap, alpha=alpha)
