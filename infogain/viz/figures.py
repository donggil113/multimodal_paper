"""The manuscript's figures.

Every function takes results objects (never raw model state), returns a
``matplotlib`` figure, and is paired with the table that backs it so the numbers
can be read without relying on colour.
"""
from __future__ import annotations

from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from infogain.viz.style import (
    CATEGORICAL,
    GRID,
    HATCH,
    INK,
    INK_MUTED,
    INK_SECONDARY,
    ROLE,
    SURFACE,
    apply_style,
    despine,
    diverging_cmap,
)


# --------------------------------------------------------------------------- #
# Fig 1: decomposition
# --------------------------------------------------------------------------- #
def fig_decomposition(df: pd.DataFrame, title: str = "",
                      figsize=(6.8, 3.6)) -> plt.Figure:
    """Marginal vs conditional value per modality, split into redundant/synergistic.

    Read left-to-right: the light bar is what the test is worth on its own; the
    dark bar is what it is still worth once everything else is in hand.  The
    hatched wedge between them is the decomposition -- blue where the context
    already had those bits, red where they only exist in combination.
    """
    apply_style()
    d = df.sort_values("conditional_bits", ascending=True).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=figsize)
    ypos = np.arange(len(d))
    h = 0.34

    ax.barh(ypos + h / 2, d["marginal_bits"], height=h, color=ROLE["marginal"],
            edgecolor=SURFACE, linewidth=1.2, label="marginal  $I_m$  (alone, on top of free data)")
    ax.barh(ypos - h / 2, d["conditional_bits"], height=h, color=ROLE["conditional"],
            edgecolor=SURFACE, linewidth=1.2,
            label="conditional  $U(m\\mid S)$  (given every other test)")

    ax.errorbar(d["marginal_bits"], ypos + h / 2,
                xerr=[d["marginal_bits"] - d["marginal_lo"],
                      d["marginal_hi"] - d["marginal_bits"]],
                fmt="none", ecolor=INK_SECONDARY, elinewidth=0.9, capsize=2)
    ax.errorbar(d["conditional_bits"], ypos - h / 2,
                xerr=[d["conditional_bits"] - d["conditional_lo"],
                      d["conditional_hi"] - d["conditional_bits"]],
                fmt="none", ecolor=INK_SECONDARY, elinewidth=0.9, capsize=2)

    span = float(max(d["marginal_hi"].max(), d["conditional_hi"].max())
                 - min(0.0, d[["marginal_bits", "conditional_bits"]].min().min()))
    pad = 0.03 * max(span, 1e-6)
    for i, r in d.iterrows():
        lo, hi = sorted((r["marginal_bits"], r["conditional_bits"]))
        syn = r["synergistic_bits"] > r["redundant_bits"]
        key = "synergistic" if syn else "redundant"
        if hi - lo > 1e-5:
            ax.add_patch(plt.Rectangle(
                (lo, i - h), hi - lo, 2 * h, facecolor=ROLE[key], alpha=0.18,
                hatch=HATCH[key], edgecolor=ROLE[key], linewidth=0.0, zorder=0))
        tag = (f"+{r['synergistic_bits']:.3f} synergistic" if syn
               else f"\u2212{r['redundant_bits']:.3f} redundant")
        ax.text(max(r["marginal_hi"], r["conditional_hi"]) + pad, i, tag,
                va="center", ha="left", fontsize=6.8, color=ROLE[key])

    ax.set_yticks(ypos)
    ax.set_yticklabels(d["modality"])
    ax.set_ylim(-0.75, len(d) - 0.25)
    ax.set_xlabel("usable information (bits)")
    left = min(0.0, float(d[["marginal_bits", "conditional_bits"]].min().min()) * 1.2)
    ax.set_xlim(left, float(max(d["marginal_hi"].max(), d["conditional_hi"].max())) + 9 * pad)
    if title:
        ax.set_title(title, loc="left")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncols=2)
    ax.grid(axis="y", visible=False)
    despine(ax)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# Fig 2: pairwise interaction map
# --------------------------------------------------------------------------- #
def fig_interaction_map(df: pd.DataFrame, modalities: Sequence[str] | None = None,
                        title: str = "", figsize=(4.6, 3.9)) -> plt.Figure:
    """Signed modality-pair interaction: blue redundant, red synergistic."""
    apply_style()
    mods = list(modalities) if modalities else sorted(set(df["a"]) | set(df["b"]))
    M = np.full((len(mods), len(mods)), np.nan)
    for _, r in df.iterrows():
        i, j = mods.index(r["a"]), mods.index(r["b"])
        M[i, j] = M[j, i] = r["interaction_bits"]

    vmax = float(np.nanmax(np.abs(M))) if np.isfinite(M).any() else 1.0
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(M, cmap=diverging_cmap(), vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(len(mods)), mods, rotation=35, ha="right")
    ax.set_yticks(range(len(mods)), mods)
    ax.grid(False)
    for i in range(len(mods)):
        for j in range(len(mods)):
            if i == j:
                ax.text(j, i, "--", ha="center", va="center", color=INK_MUTED, fontsize=8)
            elif np.isfinite(M[i, j]):
                shade = INK if abs(M[i, j]) < 0.55 * vmax else SURFACE
                ax.text(j, i, f"{M[i, j]:+.3f}", ha="center", va="center",
                        color=shade, fontsize=7.5)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label("interaction  $I(ab)-I(a)-I(b)$  (bits)")
    cb.outline.set_visible(False)
    ax.set_title(title or "redundant  <-  0  ->  synergistic", loc="left")
    despine(ax, keep=())
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# Fig 3: per-patient gain distribution
# --------------------------------------------------------------------------- #
def fig_patient_gains(results: dict[str, np.ndarray], title: str = "",
                      figsize=(7.0, 3.0)) -> plt.Figure:
    """Distribution and concentration of :math:`\\Delta_i(m)` across patients.

    The Lorenz panel carries the paper's central empirical claim: if the curves
    hugged the diagonal, the average would be the whole story and per-patient
    targeting would be pointless.
    """
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    ax = axes[0]
    for i, (name, d) in enumerate(results.items()):
        d = np.asarray(d)
        d = d[np.isfinite(d)]
        ax.hist(np.clip(d, 0, None), bins=60, histtype="step", linewidth=1.6,
                color=CATEGORICAL[i % len(CATEGORICAL)], label=name, log=True)
    ax.set_xlabel(r"per-patient gain  $\Delta_i(m\mid S)$  (bits)")
    ax.set_ylabel("patients (log)")
    ax.legend(loc="upper right")
    despine(ax)

    ax = axes[1]
    ax.plot([0, 1], [0, 1], color=INK_MUTED, linestyle=(0, (3, 3)), linewidth=1.2,
            label="uniform benefit")
    for i, (name, d) in enumerate(results.items()):
        d = np.sort(np.clip(np.asarray(d), 0, None))
        if d.sum() <= 0:
            continue
        cum = np.cumsum(d) / d.sum()
        x = np.arange(1, d.size + 1) / d.size
        ax.plot(x, cum, color=CATEGORICAL[i % len(CATEGORICAL)], label=name)
        k = int(0.8 * d.size)
        ax.annotate(f"{name}: bottom 80% of patients\nhold {cum[k]:.0%} of the benefit",
                    xy=(0.8, cum[k]), xytext=(0.06, 0.72 - 0.16 * i), fontsize=7,
                    color=INK_SECONDARY,
                    arrowprops=dict(arrowstyle="-", color=GRID, linewidth=0.8))
    ax.set_xlabel("patients ranked by gain")
    ax.set_ylabel("cumulative share of total gain")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    despine(ax)
    if title:
        fig.suptitle(title, x=0.01, ha="left", fontsize=10, fontweight="bold")
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# Fig 4: decision curves
# --------------------------------------------------------------------------- #
def fig_decision_curves(curves: dict[str, tuple[np.ndarray, np.ndarray]],
                        prevalence: float, thresholds: np.ndarray,
                        title: str = "", figsize=(5.0, 3.4)) -> plt.Figure:
    """Net benefit against risk threshold, with treat-all/treat-none references."""
    apply_style()
    fig, ax = plt.subplots(figsize=figsize)
    treat_all = prevalence - (1 - prevalence) * thresholds / np.clip(1 - thresholds, 1e-9, None)
    ax.plot(thresholds, treat_all, color=INK_MUTED, linestyle=(0, (4, 3)),
            linewidth=1.2, label="test everyone")
    ax.axhline(0.0, color=INK_MUTED, linestyle=(0, (1, 3)), linewidth=1.2,
               label="test no one")
    for i, (name, (t, nb)) in enumerate(curves.items()):
        ax.plot(t, nb, color=ROLE.get(name, CATEGORICAL[i % len(CATEGORICAL)]),
                label=name)
    lo = min(0.0, float(np.nanmin([np.nanmin(v[1]) for v in curves.values()])))
    ax.set_ylim(max(lo, -prevalence * 0.6), None)
    ax.set_xlabel("risk threshold  $t$")
    ax.set_ylabel("net benefit")
    ax.legend(loc="upper right")
    if title:
        ax.set_title(title, loc="left")
    despine(ax)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# Fig 5: reduction frontier
# --------------------------------------------------------------------------- #
def fig_reduction_frontier(frontier: pd.DataFrame, comparators: pd.DataFrame,
                           metric: str = "auroc", title: str = "",
                           figsize=(5.4, 3.6)) -> plt.Figure:
    """Tests ordered vs performance, INFOGAIN against every comparator policy."""
    apply_style()
    fig, ax = plt.subplots(figsize=figsize)
    f = frontier.sort_values("test_fraction")
    ax.plot(f["test_fraction"], f[metric], color=ROLE["infogain"], marker="o",
            markersize=3.5, label="INFOGAIN (patient-level)")

    styles = {"order_all": ("s", ROLE["order_all"]), "baseline_only": ("D", ROLE["baseline_only"])}
    for _, r in comparators.iterrows():
        name = str(r["policy"])
        if name in styles:
            mk, col = styles[name]
        elif name.startswith("risk_band"):
            mk, col = "^", ROLE["risk_band"]
        elif name.startswith("random"):
            mk, col = "v", ROLE["random"]
        else:
            mk, col = "x", CATEGORICAL[5]
        kw = ({} if mk == "x" else {"edgecolor": SURFACE, "linewidth": 0.8})
        ax.scatter(r["test_fraction"], r[metric], marker=mk, s=34, color=col,
                   zorder=5, **kw)
        if name in ("order_all", "baseline_only") or name.startswith("risk_band[0.4"):
            ax.annotate(name, (r["test_fraction"], r[metric]), fontsize=7,
                        color=INK_SECONDARY, xytext=(4, -8), textcoords="offset points")

    handles = [plt.Line2D([], [], color=ROLE["infogain"], marker="o", label="INFOGAIN"),
               plt.Line2D([], [], color=ROLE["order_all"], marker="s", linestyle="",
                          label="order all / order none"),
               plt.Line2D([], [], color=ROLE["risk_band"], marker="^", linestyle="",
                          label="risk-band heuristic"),
               plt.Line2D([], [], color=ROLE["random"], marker="v", linestyle="",
                          label="random at matched budget"),
               plt.Line2D([], [], color=CATEGORICAL[5], marker="x", linestyle="",
                          label="always one fixed test")]
    ax.legend(handles=handles, loc="lower right")
    ax.set_xlabel("fraction of orderable tests actually ordered")
    ax.set_ylabel({"auroc": "AUROC", "restricted_info_bits": "restricted usable information (bits)"}
                  .get(metric, metric))
    if title:
        ax.set_title(title, loc="left")
    despine(ax)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# Fig 6: estimator validation against ground truth
# --------------------------------------------------------------------------- #
def fig_recovery(df: pd.DataFrame, title: str = "", figsize=(6.8, 3.0),
                 floor_bits: float = 0.01) -> plt.Figure:
    """Simulation study: estimated vs known information, and recovery vs sample size.

    The right panel is restricted to subsets whose true information exceeds
    ``floor_bits``: below that, "percent recovered" divides one noise term by
    another and swings wildly in both directions, which says nothing about the
    estimator. The left panel keeps every subset, since absolute error is
    well behaved at zero.
    """
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    ax = axes[0]
    big = df[df["n"] == df["n"].max()]
    lim = float(max(big["truth_bits"].max(), big["est_bits"].max())) * 1.12 + 1e-4
    ax.plot([0, lim], [0, lim], color=INK_MUTED, linestyle=(0, (3, 3)), linewidth=1.2,
            label="perfect recovery")
    ax.errorbar(big["truth_bits"], big["est_bits"],
                yerr=[big["est_bits"] - big["ci_lo"], big["ci_hi"] - big["est_bits"]],
                fmt="o", color=CATEGORICAL[0], ecolor=INK_SECONDARY, elinewidth=0.9,
                capsize=2, markersize=4, label=f"n = {int(big['n'].max()):,}")
    if "corrected_bits" in big.columns and big["corrected_bits"].notna().any():
        ax.scatter(big["truth_bits"], big["corrected_bits"], marker="^", s=30,
                   color=CATEGORICAL[1], label="after learning-curve correction")
    ax.set_xlabel("true information (bits)")
    ax.set_ylabel("estimated information (bits)")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.legend(loc="upper left")
    despine(ax)

    ax = axes[1]
    scored = df[df.groupby("subset")["truth_bits"].transform("max") >= floor_bits]
    for i, (name, g) in enumerate(scored.groupby("subset")):
        g = g.sort_values("n")
        ax.plot(g["n"], 100 * g["est_bits"] / g["truth_bits"].clip(lower=1e-9),
                marker="o", markersize=3.5, color=CATEGORICAL[i % len(CATEGORICAL)],
                label=name.replace("demographics", "demo"))
    ax.axhline(100, color=INK_MUTED, linestyle=(0, (3, 3)), linewidth=1.2)
    ax.set_xscale("log")
    ax.set_xlabel("cohort size")
    ax.set_ylabel("% of true information recovered")
    ax.set_ylim(0, 130)
    ax.legend(loc="lower right", fontsize=5.5, ncols=2)
    despine(ax)
    if title:
        fig.suptitle(title, x=0.01, ha="left", fontsize=10, fontweight="bold")
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# Fig 7: the information <-> net-benefit identity
# --------------------------------------------------------------------------- #
def fig_identity(thresholds: np.ndarray, d_nb: np.ndarray, kl_bits: float,
                 integral_bits: float, window: tuple[float, float] = (0.02, 0.30),
                 title: str = "", figsize=(5.2, 3.3)) -> plt.Figure:
    """Theorem 3 made visible: information is the area under weighted net-benefit gain."""
    apply_style()
    fig, ax = plt.subplots(figsize=figsize)
    integrand = d_nb / np.clip(thresholds, 1e-9, None) / np.log(2)
    ax.plot(thresholds, integrand, color=CATEGORICAL[0],
            label=r"$\Delta \mathrm{NB}(t)\,/\,t\ \log 2$")
    lo, hi = window
    sel = (thresholds >= lo) & (thresholds <= hi)
    ax.fill_between(thresholds[sel], 0, integrand[sel], color=CATEGORICAL[0],
                    alpha=0.18, hatch="///", edgecolor=CATEGORICAL[0], linewidth=0)
    ax.set_xscale("log")
    ax.set_xlabel("risk threshold  $t$  (log scale)")
    ax.set_ylabel("information density (bits per unit $\\log t$)")
    ax.set_title(title or "information gain = area under this curve", loc="left")
    ax.text(0.02, 0.94,
            f"$\\Delta I$ (direct KL) = {kl_bits:.4f} bits\n"
            f"$\\int \\Delta\\mathrm{{NB}}\\,dt/t$ = {integral_bits:.4f} bits\n"
            f"shaded: clinically usable window [{lo:g}, {hi:g}]",
            transform=ax.transAxes, va="top", fontsize=7.5, color=INK_SECONDARY)
    despine(ax)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# Fig 8: Theorem 1 certified vs observed AUROC gain
# --------------------------------------------------------------------------- #
def fig_theorem1(df: pd.DataFrame, title: str = "", figsize=(4.8, 3.3)) -> plt.Figure:
    """Certified lower bound on achievable AUROC gain against what was observed."""
    apply_style()
    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(len(df))
    w = 0.36
    ax.bar(x - w / 2, df["bound"], width=w, color=ROLE["certified"],
           edgecolor=SURFACE, linewidth=1.2, hatch="///",
           label="Theorem 1 certified lower bound")
    ax.bar(x + w / 2, df["observed_gain"], width=w, color=CATEGORICAL[0],
           edgecolor=SURFACE, linewidth=1.2, label="observed AUROC gain")
    for xi, (b, o) in enumerate(zip(df["bound"], df["observed_gain"])):
        ax.text(xi - w / 2, b, f"{b:.3f}", ha="center", va="bottom", fontsize=6.5,
                color=INK_SECONDARY)
        ax.text(xi + w / 2, o, f"{o:.3f}", ha="center", va="bottom", fontsize=6.5,
                color=INK_SECONDARY)
    ax.set_xticks(x, df["modality"], rotation=25, ha="right")
    ax.set_ylabel("AUROC gain over the context model")
    ax.legend(loc="upper left")
    if title:
        ax.set_title(title, loc="left")
    ax.grid(axis="x", visible=False)
    despine(ax)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# Fig 9: patient-level heterogeneity map
# --------------------------------------------------------------------------- #
def fig_gain_by_risk(base_risk: np.ndarray, gains: dict[str, np.ndarray],
                     n_bins: int = 20, title: str = "",
                     figsize=(5.2, 3.3)) -> plt.Figure:
    """Which patients benefit: expected gain against baseline risk decile.

    Answers the objection that this reduces to risk stratification.  If a
    modality's curve were flat, ordering by baseline risk would do the same job;
    the curves are not flat and they are not shaped alike.
    """
    apply_style()
    fig, ax = plt.subplots(figsize=figsize)
    edges = np.quantile(base_risk, np.linspace(0, 1, n_bins + 1))
    centers = 0.5 * (edges[1:] + edges[:-1])
    for i, (name, g) in enumerate(gains.items()):
        means = []
        for k in range(n_bins):
            sel = (base_risk >= edges[k]) & (base_risk <= edges[k + 1])
            means.append(float(np.mean(g[sel])) if sel.any() else np.nan)
        ax.plot(centers, means, marker="o", markersize=3.2,
                color=CATEGORICAL[i % len(CATEGORICAL)], label=name)
    ax.set_xscale("log")
    ax.set_xlabel("baseline risk from free data  $P(Y=1\\mid X_{S_0})$")
    ax.set_ylabel(r"mean expected gain  $\Delta_i$  (bits)")
    ax.legend(loc="upper left", fontsize=7)
    if title:
        ax.set_title(title, loc="left")
    despine(ax)
    fig.tight_layout()
    return fig


def fig_calibration(y: np.ndarray, probs: dict[str, np.ndarray], n_bins: int = 12,
                    title: str = "", figsize=(4.4, 3.4)) -> plt.Figure:
    """Reliability diagram: the identity of Theorem 3 assumes calibrated risks."""
    apply_style()
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot([0, 1], [0, 1], color=INK_MUTED, linestyle=(0, (3, 3)), linewidth=1.2,
            label="perfect calibration")
    for i, (name, p) in enumerate(probs.items()):
        order = np.argsort(p)
        chunks = np.array_split(order, n_bins)
        xs = [float(p[c].mean()) for c in chunks if c.size]
        ys = [float(y[c].mean()) for c in chunks if c.size]
        ax.plot(xs, ys, marker="o", markersize=3.5,
                color=CATEGORICAL[i % len(CATEGORICAL)], label=name)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("predicted risk")
    ax.set_ylabel("observed frequency")
    ax.legend(loc="upper left", fontsize=7)
    if title:
        ax.set_title(title, loc="left")
    despine(ax)
    fig.tight_layout()
    return fig
