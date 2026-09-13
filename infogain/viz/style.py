"""Figure styling for the manuscript.

Colour choices follow the four-jobs rule -- categorical for identity, one-hue
sequential for magnitude, blue/red diverging with a grey midpoint for polarity --
and the categorical order below is a validated one (worst adjacent CVD Delta E 9.1,
worst adjacent normal-vision Delta E 19.6 on a near-white surface).  Three of the
slots sit below 3:1 contrast against white, so every figure that uses them also
ships direct value labels and a companion CSV; several also carry hatching, so
identity never rests on hue alone in print or for a colour-blind reader.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#8a8880"
GRID = "#e4e3df"

#: Validated categorical order (identity encoding).
CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4",
               "#008300", "#4a3aa7", "#e34948"]

#: Single-hue ramp for magnitude.
SEQUENTIAL = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
              "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281",
              "#0d366b"]

#: Diverging poles for signed quantities; grey, not a hue, at zero.
DIVERGING_LOW = "#2a78d6"     # redundant
DIVERGING_MID = "#f0efec"
DIVERGING_HIGH = "#e34948"    # synergistic

STATUS = {"good": "#0ca30c", "warning": "#fab219",
          "serious": "#ec835a", "critical": "#d03b3b"}

#: Fixed semantic roles used across figures, so a colour means the same thing
#: in every panel of the paper.
ROLE = {
    "marginal": CATEGORICAL[0],
    "conditional": CATEGORICAL[2],
    "redundant": DIVERGING_LOW,
    "synergistic": DIVERGING_HIGH,
    "infogain": CATEGORICAL[0],
    "order_all": INK_SECONDARY,
    "baseline_only": INK_MUTED,
    "risk_band": CATEGORICAL[1],
    "random": CATEGORICAL[3],
    "certified": CATEGORICAL[4],
}
HATCH = {"redundant": "///", "synergistic": "\\\\\\"}


def apply_style(base_size: float = 8.5) -> None:
    """Publication defaults: recessive chrome, real fonts, vector-safe hatching."""
    mpl.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.family": "DejaVu Sans",
        "font.size": base_size,
        "axes.titlesize": base_size + 1.5,
        "axes.titleweight": "bold",
        "axes.labelsize": base_size,
        "axes.labelcolor": INK,
        "axes.edgecolor": GRID,
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "xtick.color": INK_SECONDARY,
        "ytick.color": INK_SECONDARY,
        "xtick.labelsize": base_size - 0.5,
        "ytick.labelsize": base_size - 0.5,
        "legend.frameon": False,
        "legend.fontsize": base_size - 0.5,
        "lines.linewidth": 1.8,
        "lines.markersize": 4.5,
        "hatch.linewidth": 0.6,
        "figure.dpi": 130,
        "savefig.dpi": 400,
        "savefig.bbox": "tight",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def despine(ax, keep=("left", "bottom")) -> None:
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(side in keep)


def diverging_cmap():
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list(
        "infogain_div", [DIVERGING_LOW, DIVERGING_MID, DIVERGING_HIGH])


def sequential_cmap():
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list("infogain_seq", SEQUENTIAL)


def save(fig, path: str | Path, also_pdf: bool = True, table=None) -> Path:
    """Write the figure and, when given, the table view that backs it.

    The companion CSV is not a nicety: three palette slots fall below 3:1
    against the page, and the accessibility rule for that is an accessible
    alternative rendering of the same numbers.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".png"))
    if also_pdf:
        fig.savefig(path.with_suffix(".pdf"))
    if table is not None:
        table.to_csv(path.with_suffix(".csv"), index=False)
    plt.close(fig)
    return path
