r"""Bias correction, variance decomposition and negative controls for :math:`\hat I_{\mathcal V}`.

Two different biases afflict a :math:`\mathcal V`-information estimate and they
pull in opposite directions, so neither can be ignored:

**Learning bias (downward).**  :math:`I_{\mathcal V}` is defined by an infimum
over the family, but we only have the predictor a finite training set produced.
Every estimate is therefore a lower bound, and the shortfall shrinks as the
training set grows.  It is by far the larger of the two on realistic cohorts --
in the simulation study, recovering the synergistic term needs an order of
magnitude more data than recovering the marginal ones.  :func:`learning_curve`
and :func:`extrapolate_infinite_data` measure and correct it by refitting at
several training-set sizes and extrapolating :math:`\hat I(m) = I_\infty - a
m^{-\alpha}` to :math:`m\to\infty`.

**Evaluation bias (upward, and avoidable).**  Scoring PVI on data used for
fitting inflates it without bound.  Cross-fitting removes it by construction,
which is why nothing here ever touches in-fold predictions.

On top of both sits **estimator variance**: re-running with a new seed moves the
answer.  :func:`variance_decomposition` separates patient sampling from seed
variability so a confidence interval reflects both.

Finally, :func:`label_permutation_control` refits the whole pipeline on shuffled
labels.  Any information it reports is pure pipeline artefact, and it is the
only honest way to establish where zero is.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable, Sequence

import numpy as np
import pandas as pd
from scipy import optimize

from infogain.utils.logging import get_logger
from infogain.vinfo.core import VInfo, subset_key, v_info_from_pvi

log = get_logger("infogain.vinfo.estimators")


# --------------------------------------------------------------------------- #
# Variance decomposition
# --------------------------------------------------------------------------- #
@dataclass
class VarianceDecomposition:
    total: float
    sampling: float
    seed: float
    n_patients: int
    n_seeds: int

    @property
    def seed_share(self) -> float:
        return float(self.seed / self.total) if self.total > 0 else float("nan")

    def as_dict(self) -> dict:
        return {"var_total": self.total, "var_sampling": self.sampling,
                "var_seed": self.seed, "seed_share": self.seed_share,
                "n_patients": self.n_patients, "n_seeds": self.n_seeds}


def variance_decomposition(pvi_by_seed: np.ndarray) -> VarianceDecomposition:
    """Split the variance of :math:`\\hat I` into patient-sampling and seed parts.

    ``pvi_by_seed`` is ``(n_seeds, n_patients)``.  Treating the seed-averaged PVI
    as one sample understates uncertainty whenever different initialisations
    settle on materially different predictors -- common for the interaction terms
    that carry synergy.
    """
    arr = np.asarray(pvi_by_seed, dtype=np.float64)
    n_seeds, n = arr.shape
    seed_means = arr.mean(axis=1)
    pooled = arr.mean(axis=0)
    var_sampling = float(pooled.var(ddof=1) / n) if n > 1 else float("nan")
    var_seed = float(seed_means.var(ddof=1) / n_seeds) if n_seeds > 1 else 0.0
    return VarianceDecomposition(total=var_sampling + var_seed, sampling=var_sampling,
                                 seed=var_seed, n_patients=n, n_seeds=n_seeds)


# --------------------------------------------------------------------------- #
# Learning-curve extrapolation
# --------------------------------------------------------------------------- #
@dataclass
class LearningCurve:
    sizes: np.ndarray
    values: np.ndarray                  # (n_points,) estimates in bits
    errors: np.ndarray                  # standard errors
    subset: str
    per_point: list[dict] = field(default_factory=list)

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame({"n_train": self.sizes, "bits": self.values,
                             "se": self.errors, "subset": self.subset})


@dataclass
class Extrapolation:
    i_infinity: float
    ci_lo: float
    ci_hi: float
    alpha: float
    amplitude: float
    observed_max: float
    recovered_fraction: float
    residual_rms: float

    def as_dict(self) -> dict:
        return {"i_infinity": self.i_infinity, "ci_lo": self.ci_lo,
                "ci_hi": self.ci_hi, "alpha": self.alpha,
                "amplitude": self.amplitude, "observed_max": self.observed_max,
                "recovered_fraction": self.recovered_fraction,
                "residual_rms": self.residual_rms}


def extrapolate_infinite_data(sizes: Sequence[float], values: Sequence[float],
                              weights: Sequence[float] | None = None,
                              n_boot: int = 2000, seed: int = 0) -> Extrapolation:
    r"""Fit :math:`\hat I(m) = I_\infty - a\,m^{-\alpha}` and report :math:`I_\infty`.

    The power-law form (rather than the classical :math:`1/m` series) is used
    because the dominant bias here is *learning*, not plug-in: the shortfall of a
    trained head decays at a rate set by the problem's effective dimension, and
    forcing :math:`\alpha=1` badly over-extrapolates interaction terms.
    :math:`\alpha` is bounded to :math:`[0.15, 2]` so the fit cannot degenerate
    into a flat line through the largest point.
    """
    m = np.asarray(sizes, dtype=np.float64)
    v = np.asarray(values, dtype=np.float64)
    if m.size < 3:
        return Extrapolation(float(v[-1]), np.nan, np.nan, np.nan, np.nan,
                             float(v.max()), 1.0, 0.0)
    w = np.ones_like(v) if weights is None else 1.0 / np.clip(np.asarray(weights), 1e-9, None)

    def model(p, mm):
        return p[0] - p[1] * mm ** (-p[2])

    def resid(p, mm, vv, ww):
        return (model(p, mm) - vv) * np.sqrt(ww)

    p0 = [float(v.max()) * 1.2 + 1e-6, max(float(v.max()), 1e-3), 0.5]
    bounds = ([0.0, 0.0, 0.15], [max(1.0, 4 * float(v.max()) + 1.0), np.inf, 2.0])
    try:
        fit = optimize.least_squares(resid, p0, args=(m, v, w), bounds=bounds,
                                     max_nfev=20000)
        p = fit.x
    except Exception:  # pragma: no cover - numerical fallback
        return Extrapolation(float(v[-1]), np.nan, np.nan, np.nan, np.nan,
                             float(v.max()), 1.0, 0.0)

    rng = np.random.default_rng(seed)
    se = np.asarray(weights, dtype=np.float64) if weights is not None else np.full_like(v, 1e-4)
    boots = []
    for _ in range(n_boot):
        vb = v + rng.normal(0.0, np.clip(se, 1e-9, None))
        try:
            fb = optimize.least_squares(resid, p, args=(m, vb, w), bounds=bounds,
                                        max_nfev=4000)
            boots.append(fb.x[0])
        except Exception:  # pragma: no cover
            continue
    lo, hi = (np.quantile(boots, [0.025, 0.975]) if len(boots) > 50
              else (np.nan, np.nan))
    rms = float(np.sqrt(np.mean((model(p, m) - v) ** 2)))
    return Extrapolation(i_infinity=float(p[0]), ci_lo=float(lo), ci_hi=float(hi),
                         alpha=float(p[2]), amplitude=float(p[1]),
                         observed_max=float(v.max()),
                         recovered_fraction=float(v.max() / max(p[0], 1e-12)),
                         residual_rms=rms)


def learning_curve(fit_fn: Callable[[float, int], "object"], subset: Iterable[str],
                   fractions: Sequence[float] = (0.15, 0.3, 0.5, 0.75, 1.0),
                   seeds: Sequence[int] = (0,), n_total: int | None = None
                   ) -> LearningCurve:
    """Estimate :math:`\\hat I_{\\mathcal V}(S)` at several training-set fractions.

    ``fit_fn(fraction, seed)`` must return a
    :class:`~infogain.encoders.train.CrossFittedFamily` fitted on that fraction
    of the training data (evaluation is always on the full held-out folds, so
    only the learning bias moves along the curve).
    """
    sizes, vals, errs, per_point = [], [], [], []
    for f in fractions:
        seed_means, pvis = [], []
        for s in seeds:
            cf = fit_fn(f, s)
            pv = cf.pvi(subset)
            pvis.append(pv)
            seed_means.append(float(pv.mean()))
        pooled = np.mean(np.stack(pvis), axis=0)
        vi = v_info_from_pvi(pooled, seed_values=seed_means)
        n_eff = int(round(f * (n_total or len(pooled))))
        sizes.append(n_eff)
        vals.append(vi.value)
        errs.append(vi.se if np.isfinite(vi.se) else abs(vi.ci_hi - vi.ci_lo) / 3.92)
        per_point.append({"fraction": f, "n_train": n_eff, **vi.as_dict()})
        log.info("  learning curve: n_train=%6d  I=%.5f bits", n_eff, vi.value)
    return LearningCurve(np.array(sizes, float), np.array(vals, float),
                         np.array(errs, float), subset_key(subset), per_point)


# --------------------------------------------------------------------------- #
# Negative control
# --------------------------------------------------------------------------- #
def label_permutation_control(fit_fn: Callable[[np.ndarray, int], "object"],
                              y: np.ndarray, subsets: Sequence[Iterable[str]],
                              n_repeats: int = 5, seed: int = 0) -> pd.DataFrame:
    """Refit the whole pipeline on shuffled labels to locate the true zero.

    Cross-fitting makes :math:`\\mathbb E[\\hat I]\\le 0` under the null in theory;
    in practice fold-prevalence drift and clipping leave a small residual.
    Reporting it turns "significantly greater than zero" into a checkable claim.
    """
    rows = []
    for r in range(n_repeats):
        rng = np.random.default_rng(seed + r)
        y_perm = rng.permutation(np.asarray(y))
        cf = fit_fn(y_perm, seed + r)
        for s in subsets:
            pv = cf.pvi(s)
            vi = v_info_from_pvi(pv)
            rows.append({"repeat": r, "subset": subset_key(s), "bits": vi.value,
                         "ci_lo": vi.ci_lo, "ci_hi": vi.ci_hi})
    df = pd.DataFrame(rows)
    summary = df.groupby("subset")["bits"].agg(["mean", "std", "max"]).reset_index()
    summary.columns = ["subset", "null_mean_bits", "null_sd_bits", "null_max_bits"]
    df.attrs["summary"] = summary
    return df


# --------------------------------------------------------------------------- #
# Diagnostics
# --------------------------------------------------------------------------- #
def calibration_error(y: np.ndarray, p: np.ndarray, n_bins: int = 20) -> dict:
    """Expected and maximum calibration error on equal-frequency bins."""
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    order = np.argsort(p)
    chunks = np.array_split(order, n_bins)
    ece = mce = 0.0
    for ch in chunks:
        if ch.size == 0:
            continue
        gap = abs(y[ch].mean() - p[ch].mean())
        ece += gap * ch.size / y.size
        mce = max(mce, gap)
    return {"ece": float(ece), "mce": float(mce)}


def pvi_diagnostics(cf, subset: Iterable[str], clip: float = 8.0) -> dict:
    """Per-subset health check reported alongside every information estimate."""
    from infogain.theory.lemmas import auroc

    pv_raw = cf.pvi(subset, clip=None)
    pv = np.clip(pv_raw, -clip, clip)
    p = cf.p(subset)
    y = cf.y
    cal = calibration_error(y, p)
    return {
        "subset": subset_key(subset),
        "bits": float(pv.mean()),
        "auroc": float(auroc(y, p)),
        "brier": float(np.mean((p - y) ** 2)),
        "ece": cal["ece"], "mce": cal["mce"],
        "clipped_frac": float(np.mean(np.abs(pv_raw) > clip)),
        "pvi_negative_frac": float(np.mean(pv < 0)),
        "pvi_p95": float(np.quantile(pv, 0.95)),
        "pvi_p05": float(np.quantile(pv, 0.05)),
    }


def diagnostics_table(cf, subsets: Sequence[Iterable[str]] | None = None
                      ) -> pd.DataFrame:
    subsets = list(subsets) if subsets is not None else cf.subsets
    return pd.DataFrame([pvi_diagnostics(cf, s) for s in subsets])


# --------------------------------------------------------------------------- #
# Aggregation helper
# --------------------------------------------------------------------------- #
def aggregate(cf, subset: Iterable[str], alpha: float = 0.05,
              bias_correction: Extrapolation | None = None) -> VInfo:
    """Cohort estimate with seed variance folded in and optional bias correction."""
    pvi_seeds = cf.pvi_all_seeds(subset)
    vd = variance_decomposition(pvi_seeds)
    vi = v_info_from_pvi(pvi_seeds.mean(axis=0), alpha=alpha,
                         seed_values=list(pvi_seeds.mean(axis=1)))
    vi.var_sampling, vi.var_seed = vd.sampling, vd.seed
    vi.extra["seed_share"] = vd.seed_share
    if bias_correction is not None:
        vi.extra["i_infinity"] = bias_correction.i_infinity
        vi.extra["recovered_fraction"] = bias_correction.recovered_fraction
    return vi
