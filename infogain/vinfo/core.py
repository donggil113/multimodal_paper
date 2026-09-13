r"""Core definitions for :math:`\mathcal{V}`-usable information.

Notation follows Xu et al. (2020) and the pointwise refinement of
Ethayarajh et al. (2022).  For a predictive family :math:`\mathcal{V}` of
conditional distributions over the label :math:`Y`,

.. math::

    H_{\mathcal{V}}(Y)      &= \inf_{f\in\mathcal{V}}\ \mathbb{E}[-\log f[\varnothing](Y)] \\
    H_{\mathcal{V}}(Y|X_S)  &= \inf_{f\in\mathcal{V}}\ \mathbb{E}[-\log f[X_S](Y)] \\
    I_{\mathcal{V}}(X_S\to Y) &= H_{\mathcal{V}}(Y) - H_{\mathcal{V}}(Y|X_S).

Everything downstream is built from the **pointwise** version

.. math::

    \mathrm{pvi}_i(S) = \log_2 f_S(y_i \mid x_i^S) - \log_2 f_\varnothing(y_i),

whose sample mean estimates :math:`I_{\mathcal{V}}(X_S\to Y)` in bits.  Working
pointwise is what makes patient-level statements possible at all: the cohort
quantity is just the average of quantities that each individual patient owns.

Two properties of the estimator matter for every theorem in the paper and are
enforced here rather than assumed:

1. **Held-out evaluation.**  ``pvi`` must be computed on samples not used to fit
   :math:`f`, otherwise :math:`\hat I_{\mathcal V}` is upward biased without bound.
   The estimators in :mod:`infogain.vinfo.estimators` only ever call these
   functions with out-of-fold predictions.
2. **Masking closure.**  :math:`\mathcal{V}` must be closed under replacing a
   modality by its "absent" token, i.e. every :math:`f_S` is realisable for every
   :math:`S`.  Then :math:`S\subseteq T \Rightarrow H_{\mathcal V}(Y|X_T) \le
   H_{\mathcal V}(Y|X_S)`, so :math:`I_{\mathcal V}` is monotone on the subset
   lattice and the conditional gains of Section :math:`\S`2 are non-negative in
   population.  :func:`infogain.encoders.fusion.MaskedFusionFamily` builds such a
   family by construction.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import chain, combinations
from typing import Iterable, Iterator, Sequence

import numpy as np

LOG2 = float(np.log(2.0))
#: PVI values are clipped to +/- this many bits before averaging.  A single
#: catastrophically confident wrong prediction otherwise dominates the mean and
#: destroys every concentration bound (the range enters Bernstein linearly).
DEFAULT_PVI_CLIP = 8.0


# --------------------------------------------------------------------------- #
# Modality bookkeeping
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Modality:
    """A diagnostic modality and the resources consumed by ordering it.

    Costs are deliberately kept as three separate axes rather than collapsed
    into one number: the decision analyses in :mod:`infogain.clinical` sweep
    over their relative weighting, because a US price list, a Korean NHIS fee
    schedule and a rural clinic's turnaround constraint disagree about which
    axis binds.
    """

    name: str
    dim: int
    cost_usd: float = 0.0
    turnaround_min: float = 0.0
    invasiveness: float = 0.0  # 0 = non-invasive, 1 = major procedure
    always_available: bool = False  # e.g. vitals: never a decision variable

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return self.name


@dataclass
class ModalitySpec:
    """Ordered registry of the modalities in an experiment."""

    modalities: list[Modality]

    def __post_init__(self) -> None:
        names = [m.name for m in self.modalities]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate modality names: {names}")

    @property
    def names(self) -> list[str]:
        return [m.name for m in self.modalities]

    @property
    def orderable(self) -> list[str]:
        """Modalities a clinician can decide to order (excludes baseline data)."""
        return [m.name for m in self.modalities if not m.always_available]

    @property
    def baseline(self) -> frozenset[str]:
        """Modalities assumed present for every patient."""
        return frozenset(m.name for m in self.modalities if m.always_available)

    def __len__(self) -> int:
        return len(self.modalities)

    def __iter__(self) -> Iterator[Modality]:
        return iter(self.modalities)

    def __getitem__(self, key: str | int) -> Modality:
        if isinstance(key, int):
            return self.modalities[key]
        for m in self.modalities:
            if m.name == key:
                return m
        raise KeyError(key)

    def index(self, name: str) -> int:
        return self.names.index(name)

    def dims(self) -> dict[str, int]:
        return {m.name: m.dim for m in self.modalities}

    def costs(self) -> dict[str, float]:
        return {m.name: m.cost_usd for m in self.modalities}


# --------------------------------------------------------------------------- #
# Subset lattice helpers
# --------------------------------------------------------------------------- #
def subset_key(subset: Iterable[str]) -> str:
    """Canonical, human-readable key for a modality subset."""
    s = sorted(subset)
    return "+".join(s) if s else "EMPTY"


def parse_key(key: str) -> frozenset[str]:
    return frozenset() if key == "EMPTY" else frozenset(key.split("+"))


def iter_subsets(names: Sequence[str], include_empty: bool = True,
                 max_size: int | None = None) -> Iterator[frozenset[str]]:
    """Enumerate the subset lattice, small sets first."""
    lo = 0 if include_empty else 1
    hi = len(names) if max_size is None else min(max_size, len(names))
    for r in range(lo, hi + 1):
        for c in combinations(sorted(names), r):
            yield frozenset(c)


def powerset(names: Sequence[str]) -> Iterator[frozenset[str]]:
    return (frozenset(c) for c in
            chain.from_iterable(combinations(sorted(names), r)
                                for r in range(len(names) + 1)))


# --------------------------------------------------------------------------- #
# Pointwise V-information
# --------------------------------------------------------------------------- #
def label_logprob(probs: np.ndarray, y: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    r"""Natural log-probability assigned to the realised label.

    ``probs`` is ``(n,)`` for binary problems (P(Y=1)) or ``(n, K)`` for
    multiclass.  Probabilities are clipped to ``[eps, 1-eps]`` before the log;
    ``eps=1e-7`` corresponds to a ~23.3-bit worst case, which the PVI clip then
    caps again.
    """
    probs = np.asarray(probs, dtype=np.float64)
    y = np.asarray(y)
    if probs.ndim == 1:
        p = np.clip(probs, eps, 1.0 - eps)
        return np.log(np.where(y == 1, p, 1.0 - p))
    p = np.clip(probs, eps, 1.0)
    p = p / p.sum(axis=1, keepdims=True)
    return np.log(p[np.arange(len(y)), y.astype(int)])


def pointwise_v_information(logp_model: np.ndarray, logp_null: np.ndarray,
                            clip: float | None = DEFAULT_PVI_CLIP) -> np.ndarray:
    r"""PVI in **bits**: :math:`\log_2 f_S(y_i|x_i) - \log_2 f_\varnothing(y_i)`.

    Positive means the modality subset made *this* patient's true outcome more
    predictable than the prevalence alone; negative means the model was actively
    misled by that patient's data, which is diagnostic information in its own
    right (see the PVI-outlier analysis in the paper).
    """
    pvi = (np.asarray(logp_model, dtype=np.float64)
           - np.asarray(logp_null, dtype=np.float64)) / LOG2
    if clip is not None:
        pvi = np.clip(pvi, -clip, clip)
    return pvi


# --------------------------------------------------------------------------- #
# Estimate container
# --------------------------------------------------------------------------- #
@dataclass
class VInfo:
    """A :math:`\\mathcal V`-information estimate with uncertainty, in bits."""

    value: float
    se: float = float("nan")
    ci_lo: float = float("nan")
    ci_hi: float = float("nan")
    n: int = 0
    method: str = "plugin"
    #: split of the total variance into sampling vs. estimator (seed) components
    var_sampling: float = float("nan")
    var_seed: float = float("nan")
    extra: dict = field(default_factory=dict)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (f"VInfo({self.value:.4f} bits, 95% CI "
                f"[{self.ci_lo:.4f}, {self.ci_hi:.4f}], n={self.n}, {self.method})")

    def as_dict(self) -> dict:
        return {
            "value": self.value, "se": self.se, "ci_lo": self.ci_lo,
            "ci_hi": self.ci_hi, "n": self.n, "method": self.method,
            "var_sampling": self.var_sampling, "var_seed": self.var_seed,
            **{f"extra.{k}": v for k, v in self.extra.items()},
        }

    @property
    def significant(self) -> bool:
        """CI excludes zero from below (the only direction that is a claim)."""
        return bool(self.ci_lo > 0.0)


def v_info_from_pvi(pvi: np.ndarray, alpha: float = 0.05,
                    method: str = "bootstrap", clip: float = DEFAULT_PVI_CLIP,
                    seed_values: Sequence[float] | None = None) -> VInfo:
    """Aggregate PVI values into a cohort estimate with a finite-sample CI.

    ``seed_values`` (per-seed cohort means) are used to add the estimator's own
    training variance to the sampling variance; ignoring it is the single most
    common way multimodal-information papers overstate significance.
    """
    from infogain.vinfo.bounds import empirical_bernstein_ci, bootstrap_ci

    pvi = np.asarray(pvi, dtype=np.float64)
    n = int(pvi.size)
    if n == 0:
        return VInfo(float("nan"), n=0, method=method)
    value = float(pvi.mean())
    var_sampling = float(pvi.var(ddof=1) / n) if n > 1 else float("nan")

    var_seed = float("nan")
    if seed_values is not None and len(seed_values) > 1:
        sv = np.asarray(seed_values, dtype=np.float64)
        var_seed = float(sv.var(ddof=1) / len(sv))

    total_var = np.nansum([var_sampling, var_seed])
    se = float(np.sqrt(total_var)) if total_var > 0 else float("nan")

    if method == "bernstein":
        lo, hi = empirical_bernstein_ci(pvi, alpha=alpha, rng_width=2.0 * clip)
    elif method == "bootstrap":
        lo, hi = bootstrap_ci(pvi, alpha=alpha)
    elif method == "normal":
        z = 1.959963984540054
        lo, hi = value - z * se, value + z * se
    else:
        raise ValueError(f"unknown CI method {method!r}")

    # Widen an (asymptotic or finite-sample) interval by the seed component,
    # which is orthogonal to the sampling fluctuation the interval captures.
    if np.isfinite(var_seed) and var_seed > 0:
        pad = 1.959963984540054 * np.sqrt(var_seed)
        lo, hi = lo - pad, hi + pad

    return VInfo(value=value, se=se, ci_lo=float(lo), ci_hi=float(hi), n=n,
                 method=method, var_sampling=var_sampling, var_seed=var_seed)


def paired_gain(pvi_large: np.ndarray, pvi_small: np.ndarray,
                alpha: float = 0.05, method: str = "bootstrap",
                clip: float = DEFAULT_PVI_CLIP) -> VInfo:
    r"""Conditional gain :math:`U_{\mathcal V}(m\mid S)` as a *paired* estimate.

    Both PVI vectors share the null term, so the difference is
    :math:`\log_2 f_{S\cup m}(y_i|\cdot) - \log_2 f_S(y_i|\cdot)`: the null model
    cancels exactly and the pairing removes the (large) patient-level variance
    that dominates each term separately.  Estimating the gain as a difference of
    two independently-computed cohort means instead would inflate the standard
    error by roughly an order of magnitude at realistic cohort sizes.
    """
    diff = np.asarray(pvi_large, dtype=np.float64) - np.asarray(pvi_small, dtype=np.float64)
    out = v_info_from_pvi(diff, alpha=alpha, method=method, clip=clip)
    out.method = f"paired-{out.method}"
    return out
