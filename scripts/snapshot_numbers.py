#!/usr/bin/env python3
"""Flatten the results tree into one scalar dictionary, for before/after diffing.

A default-architecture switch changes every estimate in the paper.  Claiming it
"changes nothing material" is only checkable if both states are recorded in the
same shape, so this walks the results tree and emits every scalar it finds under
a stable dotted key.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def walk_json(obj, prefix, out):
    if isinstance(obj, dict):
        for k, v in obj.items():
            walk_json(v, f"{prefix}.{k}" if prefix else str(k), out)
    elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
        out[prefix] = float(obj)
    elif isinstance(obj, bool):
        out[prefix] = bool(obj)


#: (csv relative path, key columns, value columns)
TABLES = [
    ("tables/decomposition.csv", ["context_kind", "modality"],
     ["marginal_bits", "conditional_bits", "redundant_bits", "synergistic_bits",
      "p_conditional"]),
    ("tables/shapley.csv", ["modality"], ["shapley_bits", "share"]),
    ("tables/lattice.csv", ["subset"], ["bits"]),
    ("tables/interaction_map.csv", ["a", "b"], ["interaction_bits", "regime"]),
    ("tables/patient_gain_summary.csv", ["modality"],
     ["mean_bits", "p99_bits", "gini"]),
    ("tables/gain_consistency.csv", ["modality"],
     ["prospective_mean_bits", "retrospective_mean_bits", "spearman"]),
]


def snapshot(results: Path, cohort: str) -> dict:
    out: dict = {}
    thy = results / "theory" / "verification.json"
    if thy.exists():
        for r in json.loads(thy.read_text())["reports"]:
            out[f"theory.{r['name']}.worst_slack"] = float(r["worst_slack"])
            out[f"theory.{r['name']}.violated"] = int(r["violated"])
    sim = results / "simulation" / "summary.json"
    if sim.exists():
        walk_json(json.loads(sim.read_text()), "simulation", out)
    for oc_dir in sorted((results / cohort).glob("*")):
        if not oc_dir.is_dir():
            continue
        sj = oc_dir / "summary.json"
        if sj.exists():
            walk_json(json.loads(sj.read_text()), f"{oc_dir.name}", out)
        for rel, keys, vals in TABLES:
            f = oc_dir / rel
            if not f.exists():
                continue
            df = pd.read_csv(f)
            if not set(keys) <= set(df.columns):
                continue
            for _, row in df.iterrows():
                tag = ".".join(str(row[k]) for k in keys)
                for v in vals:
                    if v in df.columns and pd.notna(row[v]):
                        val = row[v]
                        out[f"{oc_dir.name}.{Path(rel).stem}.{tag}.{v}"] = (
                            float(val) if not isinstance(val, str) else val)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", default="results")
    ap.add_argument("--cohort-name", default="cohort_synthetic")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    snap = snapshot(Path(a.results), a.cohort_name)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(snap, indent=1, sort_keys=True))
    print(f"wrote {a.out}: {len(snap)} scalars")


if __name__ == "__main__":
    main()
