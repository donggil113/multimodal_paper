#!/usr/bin/env python3
"""Emit the supplementary LaTeX tables from the results tree.

Each table is written as a bare ``tabular`` to ``paper/tables/`` and pulled in by
``\\input`` from ``supplementary.tex``, so the manuscript never carries a
transcribed number.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


_ESCAPES = {
    "\\": r"\textbackslash{}", "{": r"\{", "}": r"\}", "$": r"\$",
    "&": r"\&", "#": r"\#", "_": r"\_", "%": r"\%", "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}", "<": r"\textless{}", ">": r"\textgreater{}",
}


def esc(s) -> str:
    """Escape a cell for LaTeX text mode.

    Claim strings such as ``I >= 2 pi (1-pi) TV^2`` carry ``>`` and ``^``, which
    are not merely cosmetic problems -- unescaped they abort the build -- so the
    escape table covers every character TeX treats specially, not the usual
    four.
    """
    out = []
    for ch in str(s):
        out.append(_ESCAPES.get(ch, ch))
    return "".join(out)


def to_tabular(df: pd.DataFrame, floatfmt: str = "%.4f",
               align: str | None = None) -> str:
    cols = list(df.columns)
    align = align or ("l" + "r" * (len(cols) - 1))
    lines = [r"\begin{tabular}{" + align + "}", r"\toprule",
             " & ".join(r"\textbf{" + esc(c) + "}" for c in cols) + r" \\",
             r"\midrule"]
    for _, row in df.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            if isinstance(v, float):
                cells.append("--" if pd.isna(v) else floatfmt % v)
            elif isinstance(v, (int,)) and not isinstance(v, bool):
                cells.append(f"{v:d}")
            else:
                cells.append(esc(v))
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def write(df: pd.DataFrame | None, path: Path, **kw) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if df is None or not len(df):
        path.write_text(r"\emph{(stage not yet run)}" + "\n")
        print(f"  {path.name}: pending")
        return
    path.write_text(to_tabular(df, **kw) + "\n")
    print(f"  {path.name}: {len(df)} rows")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", default="results")
    ap.add_argument("--cohort-name", default="cohort_synthetic")
    ap.add_argument("--primary", default="mortality_30d")
    ap.add_argument("--out", default="paper/tables")
    args = ap.parse_args()

    R, out = Path(args.results), Path(args.out)
    base = R / args.cohort_name / args.primary
    print("writing supplementary tables:")

    # S1 theory
    thy = R / "theory" / "verification.json"
    if thy.exists():
        rep = json.loads(thy.read_text())["reports"]
        df = pd.DataFrame(rep)[["name", "n_trials", "worst_slack", "violated", "holds"]]
        df.columns = ["claim", "trials", "worst slack", "violations", "holds"]
        df["holds"] = df["holds"].map({True: "yes", False: "NO"})
        df["worst slack"] = df["worst slack"].map(lambda v: f"{v:.2e}")
        write(df, out / "theory_table.tex", align="lrrrl")
    else:
        write(None, out / "theory_table.tex")

    # S2 recovery
    rec = R / "simulation" / "tables" / "recovery.csv"
    if rec.exists():
        df = pd.read_csv(rec)
        keep = ["n", "subset", "truth_bits", "est_bits", "ci_lo", "ci_hi",
                "recovered", "corrected_bits"]
        df = df[[c for c in keep if c in df.columns]].copy()
        df["subset"] = df["subset"].str.replace("demographics", "demo")
        df["recovered"] = 100 * df["recovered"]
        df.columns = ["n", "subset", "true bits", "est bits", "CI lo", "CI hi",
                      "recovered \\%", "corrected"][:len(df.columns)]
        write(df, out / "recovery_table.tex", align="rl" + "r" * (len(df.columns) - 2))
    else:
        write(None, out / "recovery_table.tex")

    # S3 regime recovery
    reg = R / "simulation" / "tables" / "regime_recovery.csv"
    if reg.exists():
        df = pd.read_csv(reg)
        keep = ["n", "modality", "true_marginal", "est_marginal", "true_conditional",
                "est_conditional", "true_regime", "est_regime", "regime_correct"]
        df = df[[c for c in keep if c in df.columns]]
        df = df.rename(columns={"true_marginal": "true $I_m$", "est_marginal": "est $I_m$",
                                "true_conditional": "true $U$", "est_conditional": "est $U$",
                                "true_regime": "true regime", "est_regime": "est regime",
                                "regime_correct": "correct"})
        align = ("rl" + "r" * 4 + "llc") if "n" in df.columns else ("l" + "r" * 4 + "llc")
        assert len(align) == len(df.columns), (align, list(df.columns))
        write(df, out / "regime_table.tex", align=align)
    else:
        write(None, out / "regime_table.tex")

    # S3b: first cohort size at which each regime is called correctly
    fc = R / "simulation" / "tables" / "regime_first_correct.csv"
    if fc.exists():
        df = pd.read_csv(fc)
        df.columns = [c.replace("_", " ") for c in df.columns]
        write(df, out / "regime_first_correct_table.tex")
    else:
        write(None, out / "regime_first_correct_table.tex")

    # synergy power grid
    ab = R / "simulation" / "tables" / "synergy_power.csv"
    if ab.exists():
        df = pd.read_csv(ab)
        keep = ["n", "complete", "events_with_both", "modality", "true_conditional",
                "est_conditional", "true_regime", "est_regime"]
        df = df[[c for c in keep if c in df.columns]]
        df.columns = [c.replace("_", " ") for c in df.columns]
        write(df, out / "power_table.tex")
    else:
        write(None, out / "power_table.tex")

    tr = R / "simulation" / "tables" / "synergy_power_tree.csv"
    if tr.exists():
        df = pd.read_csv(tr)
        keep = ["n", "complete", "events_with_both", "modality",
                "true_conditional", "est_conditional"]
        df = df[[c for c in keep if c in df.columns]]
        df.columns = [c.replace("_", " ") for c in df.columns]
        write(df, out / "tree_reference_table.tex")
    else:
        write(None, out / "tree_reference_table.tex")

    # S4 prospective vs retrospective
    cons = base / "tables" / "gain_consistency.csv"
    if cons.exists():
        df = pd.read_csv(cons)
        df.columns = [c.replace("_", " ") for c in df.columns]
        write(df, out / "consistency_table.tex")
    else:
        write(None, out / "consistency_table.tex")

    # S5 overlap
    ovl = base / "tables" / "overlap_diagnostics.csv"
    if ovl.exists():
        df = pd.read_csv(ovl)
        keep = ["modality", "order_rate", "propensity_auroc", "frac_off_support",
                "propensity_p01", "propensity_p99", "ess_ratio"]
        df = df[[c for c in keep if c in df.columns]]
        df.columns = [c.replace("_", " ") for c in df.columns]
        write(df, out / "overlap_table.tex")
    else:
        write(None, out / "overlap_table.tex")

    # S6 lattice
    lat = base / "tables" / "lattice.csv"
    if lat.exists():
        df = pd.read_csv(lat)
        df["subset"] = df["subset"].str.replace("demographics", "demo")
        df = df[["subset", "size", "bits", "ci_lo", "ci_hi"]]
        df.columns = ["subset", "$|S|$", "bits", "CI lo", "CI hi"]
        write(df, out / "lattice_table.tex", align="lrrrr")
    else:
        write(None, out / "lattice_table.tex")

    # main-text table: decomposition across endpoints
    rows = []
    for oc_dir in sorted((R / args.cohort_name).glob("*")):
        f = oc_dir / "tables" / "decomposition.csv"
        if not f.exists():
            continue
        d = pd.read_csv(f)
        d = d[d["context_kind"] == "leave_one_out"]
        for _, r in d.iterrows():
            rows.append({"endpoint": oc_dir.name, "modality": r["modality"],
                         "marginal": r["marginal_bits"],
                         "conditional": r["conditional_bits"],
                         "redundant": r["redundant_bits"],
                         "synergistic": r["synergistic_bits"],
                         "p": r["p_conditional"]})
    write(pd.DataFrame(rows) if rows else None, out / "decomposition_table.tex",
          align="llrrrrr")

    # main-text table: policy comparison
    comp = base / "tables" / "policy_comparators.csv"
    if comp.exists():
        df = pd.read_csv(comp)
        keep = ["policy", "tests_per_patient", "mean_cost", "auroc",
                "info_gap_vs_all_bits"]
        keep += [c for c in df.columns if c.startswith("nb@")][:2]
        df = df[[c for c in keep if c in df.columns]]
        df.columns = [c.replace("_", " ") for c in df.columns]
        write(df, out / "policy_table.tex")
    else:
        write(None, out / "policy_table.tex")


if __name__ == "__main__":
    main()
