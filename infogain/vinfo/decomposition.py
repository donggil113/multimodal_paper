r"""Context-relative decomposition of multimodal usable information.

Partial information decomposition is the natural language for "is this test
redundant or does it add something", but the classical PID lattice needs a
redundancy functional that its own axioms do not pin down, and the competing
choices disagree on real data.  Rather than pick a side, we define the
decomposition *relative to the decision context* -- which is what a clinician
actually faces -- and every term becomes an identified, estimable quantity.

Fix a free baseline :math:`S_0` (demographics, vitals: acquired for everyone,
never a decision) and a context :math:`S\supseteq S_0` of tests already in hand.
For a candidate modality :math:`m`:

.. math::
    \text{marginal}\quad & I_m := I_{\mathcal V}(X_{S_0\cup m}\to Y) - I_{\mathcal V}(X_{S_0}\to Y) \\
    \text{conditional}\quad & U(m\mid S) := I_{\mathcal V}(X_{S\cup m}\to Y) - I_{\mathcal V}(X_S\to Y) \\
    \text{redundant}\quad & R(m\mid S) := \big(I_m - U(m\mid S)\big)_+ \\
    \text{synergistic}\quad & \mathrm{Syn}(m\mid S) := \big(U(m\mid S) - I_m\big)_+

so that :math:`U(m\mid S) = I_m - R(m\mid S) + \mathrm{Syn}(m\mid S)` exactly,
with :math:`R\cdot\mathrm{Syn}=0`.  Redundant bits are the ones the context
already had -- Theorem 2 turns them into a lower bound on wasted expenditure.
Synergistic bits exist only in combination -- Theorem 1 turns them into a lower
bound on achievable discrimination.

Two refinements matter in practice:

* **Everything is estimated pairwise.**  Both terms of every difference share
  the same patients and the same null model, so differencing per patient before
  averaging removes the patient-level variance that dominates each term alone.
* **Order dependence is handled by Shapley averaging.**  :math:`U(m\mid S)`
  depends on which tests you happen to have already; :func:`shapley_information`
  averages over all acquisition orders, giving the unique attribution satisfying
  efficiency, symmetry, null-player and additivity.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from math import factorial
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from infogain.vinfo.bounds import paired_permutation_pvalue
from infogain.vinfo.core import VInfo, subset_key, v_info_from_pvi


# --------------------------------------------------------------------------- #
# Single-modality decomposition
# --------------------------------------------------------------------------- #
@dataclass
class ModalityDecomposition:
    modality: str
    baseline: frozenset[str]
    context: frozenset[str]
    marginal: VInfo
    conditional: VInfo
    redundant: float
    synergistic: float
    redundancy_fraction: float
    p_conditional: float
    n: int

    def as_dict(self) -> dict:
        return {
            "modality": self.modality,
            "baseline": subset_key(self.baseline),
            "context": subset_key(self.context),
            "marginal_bits": self.marginal.value,
            "marginal_lo": self.marginal.ci_lo, "marginal_hi": self.marginal.ci_hi,
            "conditional_bits": self.conditional.value,
            "conditional_lo": self.conditional.ci_lo,
            "conditional_hi": self.conditional.ci_hi,
            "redundant_bits": self.redundant,
            "synergistic_bits": self.synergistic,
            "redundancy_fraction": self.redundancy_fraction,
            "p_conditional": self.p_conditional,
            "n": self.n,
        }


def _paired(cf, big: Iterable[str], small: Iterable[str],
            rows: np.ndarray | None = None, alpha: float = 0.05) -> VInfo:
    """Paired estimate of ``I(big) - I(small)`` with seed variance folded in."""
    diff = cf.pvi(big, rows=rows) - cf.pvi(small, rows=rows)
    seed_diffs = [float((cf.pvi(big, s, rows=rows) - cf.pvi(small, s, rows=rows)).mean())
                  for s in range(cf.n_seeds)]
    out = v_info_from_pvi(diff, alpha=alpha, seed_values=seed_diffs)
    out.method = f"paired-{out.method}"
    return out


def decompose_modality(cf, modality: str, context: Iterable[str] | None = None,
                       baseline: Iterable[str] | None = None,
                       rows: np.ndarray | None = None, alpha: float = 0.05,
                       n_perm: int = 5000) -> ModalityDecomposition:
    """Marginal / conditional / redundant / synergistic split for one modality."""
    base = frozenset(baseline) if baseline is not None else cf.baseline
    if context is None:
        context = frozenset(cf.names) - {modality}
    ctx = frozenset(context) - {modality}
    if not cf.has(base | {modality}) or not cf.has(ctx | {modality}):
        raise KeyError(f"missing fitted subsets for {modality!r}; "
                       "extend the `subsets` passed to fit_family")

    marginal = _paired(cf, base | {modality}, base, rows, alpha)
    conditional = _paired(cf, ctx | {modality}, ctx, rows, alpha)
    red = max(marginal.value - conditional.value, 0.0)
    syn = max(conditional.value - marginal.value, 0.0)
    p = paired_permutation_pvalue(cf.pvi(ctx | {modality}, rows=rows),
                                  cf.pvi(ctx, rows=rows), n_perm=n_perm)
    denom = max(abs(marginal.value), 1e-9)
    return ModalityDecomposition(
        modality=modality, baseline=base, context=ctx, marginal=marginal,
        conditional=conditional, redundant=red, synergistic=syn,
        redundancy_fraction=float(red / denom), p_conditional=float(p),
        n=int(len(cf.y) if rows is None else len(rows)))


def decomposition_table(cf, context: Iterable[str] | None = None,
                        baseline: Iterable[str] | None = None,
                        rows: np.ndarray | None = None,
                        alpha: float = 0.05, n_perm: int = 5000) -> pd.DataFrame:
    """Decomposition for every orderable modality, leave-one-out by default.

    ``n_perm`` is exposed so a diagnostic sweep can skip the permutation test,
    which is the most expensive part of this function and answers a question
    ("is the gain distinguishable from zero") that a sweep comparing estimators
    does not ask.
    """
    base = frozenset(baseline) if baseline is not None else cf.baseline
    orderable = [m for m in cf.names if m not in base]
    out = []
    for m in orderable:
        ctx = (frozenset(cf.names) - {m}) if context is None else frozenset(context) - {m}
        out.append(decompose_modality(cf, m, ctx, base, rows, alpha, n_perm).as_dict())
    return pd.DataFrame(out)


# --------------------------------------------------------------------------- #
# Pairwise interaction map
# --------------------------------------------------------------------------- #
def pairwise_interaction_map(cf, baseline: Iterable[str] | None = None,
                             rows: np.ndarray | None = None,
                             alpha: float = 0.05) -> pd.DataFrame:
    r"""Signed interaction :math:`\Delta_{ab} = I(ab) - I(a) - I(b)`, all pairs.

    Everything is measured on top of the free baseline.  Negative means the pair
    is *redundant* -- the two tests overlap and the second buys less than it
    would alone; positive means *synergistic* -- the pair is worth more than the
    sum of its parts.  This single signed number per pair is the object the
    modality maps in the paper are drawn from, and unlike a PID atom it needs no
    contested axiom to define.
    """
    base = frozenset(baseline) if baseline is not None else cf.baseline
    orderable = [m for m in cf.names if m not in base]
    rowsl = []
    for a, b in combinations(orderable, 2):
        ga = cf.pvi(base | {a}, rows=rows) - cf.pvi(base, rows=rows)
        gb = cf.pvi(base | {b}, rows=rows) - cf.pvi(base, rows=rows)
        gab = cf.pvi(base | {a, b}, rows=rows) - cf.pvi(base, rows=rows)
        inter = gab - ga - gb
        seed_vals = [float((cf.pvi(base | {a, b}, s, rows=rows)
                            - cf.pvi(base | {a}, s, rows=rows)
                            - cf.pvi(base | {b}, s, rows=rows)
                            + cf.pvi(base, s, rows=rows)).mean())
                     for s in range(cf.n_seeds)]
        vi = v_info_from_pvi(inter, alpha=alpha, seed_values=seed_vals)
        rowsl.append({
            "a": a, "b": b,
            "I_a": float(ga.mean()), "I_b": float(gb.mean()), "I_ab": float(gab.mean()),
            "interaction_bits": vi.value, "ci_lo": vi.ci_lo, "ci_hi": vi.ci_hi,
            "regime": ("synergistic" if vi.ci_lo > 0 else
                       "redundant" if vi.ci_hi < 0 else "indeterminate"),
            "redundancy_bits": max(-vi.value, 0.0),
            "synergy_bits": max(vi.value, 0.0),
        })
    return pd.DataFrame(rowsl)


# --------------------------------------------------------------------------- #
# Shapley usable information
# --------------------------------------------------------------------------- #
def _shapley_weights(n: int) -> dict[int, float]:
    return {k: factorial(k) * factorial(n - k - 1) / factorial(n) for k in range(n)}


def shapley_information(cf, baseline: Iterable[str] | None = None,
                        rows: np.ndarray | None = None, alpha: float = 0.05
                        ) -> pd.DataFrame:
    r"""Shapley attribution of usable information over acquisition orders.

    :math:`\varphi_m = \sum_{T\subseteq M\setminus m}
    \frac{|T|!\,(|M|-|T|-1)!}{|M|!}\,[v(T\cup m)-v(T)]` with
    :math:`v(T)=I_{\mathcal V}(X_{S_0\cup T}\to Y)`.

    Because :math:`v` is itself a sample mean of PVI, the whole Shapley value is
    a mean of *per-patient* Shapley contributions, so the same computation yields
    a patient-level attribution and a bootstrap interval for free.  Efficiency
    guarantees the values sum to the total usable information of the full panel,
    which makes the resulting bar chart honest: no modality can be credited with
    bits the panel does not contain.
    """
    base = frozenset(baseline) if baseline is not None else cf.baseline
    orderable = [m for m in cf.names if m not in base]
    n = len(orderable)
    w = _shapley_weights(n)

    pvi_cache: dict[str, np.ndarray] = {}

    def v(T: frozenset[str]) -> np.ndarray:
        k = subset_key(T)
        if k not in pvi_cache:
            pvi_cache[k] = cf.pvi(base | T, rows=rows) - cf.pvi(base, rows=rows)
        return pvi_cache[k]

    n_pat = len(cf.y) if rows is None else len(rows)
    phi = {m: np.zeros(n_pat) for m in orderable}
    for m in orderable:
        others = [o for o in orderable if o != m]
        for r in range(len(others) + 1):
            for T in combinations(others, r):
                Ts = frozenset(T)
                phi[m] += w[r] * (v(Ts | {m}) - v(Ts))

    total = v(frozenset(orderable))
    rowsl = []
    for m in orderable:
        vi = v_info_from_pvi(phi[m], alpha=alpha)
        rowsl.append({"modality": m, "shapley_bits": vi.value,
                      "ci_lo": vi.ci_lo, "ci_hi": vi.ci_hi,
                      "share": float(vi.value / max(total.mean(), 1e-12))})
    df = pd.DataFrame(rowsl)
    df.attrs["total_bits"] = float(total.mean())
    df.attrs["efficiency_residual"] = float(df["shapley_bits"].sum() - total.mean())
    df.attrs["per_patient"] = phi
    return df


# --------------------------------------------------------------------------- #
# Lattice table and monotone projection
# --------------------------------------------------------------------------- #
def lattice_table(cf, rows: np.ndarray | None = None,
                  alpha: float = 0.05) -> pd.DataFrame:
    out = []
    for s in cf.subsets:
        vi = v_info_from_pvi(cf.pvi(s, rows=rows), alpha=alpha,
                             seed_values=cf.seed_means(s, rows=rows))
        out.append({"subset": subset_key(s), "size": len(s), "bits": vi.value,
                    "ci_lo": vi.ci_lo, "ci_hi": vi.ci_hi, "n": vi.n})
    return pd.DataFrame(out).sort_values(["size", "subset"]).reset_index(drop=True)


def monotone_projection(values: dict[str, float], subsets: Sequence[frozenset[str]],
                        max_iter: int = 2000, tol: float = 1e-10) -> dict[str, float]:
    r"""Project lattice values onto the monotone cone :math:`S\subseteq T\Rightarrow v_S\le v_T`.

    Masking closure makes :math:`I_{\mathcal V}` monotone in population, but a
    finite sample and an imperfectly optimised head can violate it by a few
    thousandths of a bit.  Reporting the raw values invites the objection that
    the estimator is incoherent; silently clipping them hides real estimation
    error.  We do neither: Dykstra's cyclic projection onto the covering-relation
    half-spaces returns the closest monotone vector in Euclidean norm, and the
    displacement is reported as a diagnostic alongside the raw values.
    """
    keys = [subset_key(s) for s in subsets]
    pos = {k: i for i, k in enumerate(keys)}
    v = np.array([values[k] for k in keys], dtype=np.float64)

    edges = [(i, j) for i, s in enumerate(subsets) for j, t in enumerate(subsets)
             if i != j and len(t) == len(s) + 1 and s < t]   # require x[i] <= x[j]
    if not edges:
        return dict(values)

    # Dykstra's algorithm: cyclic projection with per-constraint correction terms.
    x = v.copy()
    z = {e: np.zeros(2) for e in edges}
    for _ in range(max_iter):
        shift = 0.0
        for e in edges:
            i, j = e
            yi, yj = x[i] + z[e][0], x[j] + z[e][1]
            if yi > yj:                       # project onto {x_i <= x_j}
                mid = 0.5 * (yi + yj)
                ni, nj = mid, mid
            else:
                ni, nj = yi, yj
            z[e][0], z[e][1] = yi - ni, yj - nj
            shift = max(shift, abs(ni - x[i]), abs(nj - x[j]))
            x[i], x[j] = ni, nj
        if shift < tol:
            break
    return {k: float(x[pos[k]]) for k in keys}


def monotonicity_report(cf, rows: np.ndarray | None = None) -> dict:
    """How badly (if at all) the estimated lattice violates monotonicity."""
    raw = {subset_key(s): float(cf.pvi(s, rows=rows).mean()) for s in cf.subsets}
    proj = monotone_projection(raw, cf.subsets)
    disp = np.array([proj[k] - raw[k] for k in raw])
    viol = 0
    worst = 0.0
    for s in cf.subsets:
        for t in cf.subsets:
            if s < t and raw[subset_key(s)] > raw[subset_key(t)] + 1e-12:
                viol += 1
                worst = max(worst, raw[subset_key(s)] - raw[subset_key(t)])
    return {"n_violations": viol, "worst_violation_bits": worst,
            "max_projection_shift_bits": float(np.abs(disp).max()),
            "mean_abs_shift_bits": float(np.abs(disp).mean()),
            "raw": raw, "projected": proj}
