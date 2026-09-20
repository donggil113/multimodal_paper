#!/usr/bin/env python3
"""Diff two results snapshots and flag what the default-architecture switch moved.

Switching the fusion head changes every estimate in the paper. The claim that it
changes nothing *material* is only worth making if the change is enumerated, and
the enumeration has to separate three cases that a mean absolute difference
would blur together:

* a **sign flip** -- a modality called redundant now called synergistic, or a
  gain that changed direction. These are the ones that would rewrite a sentence,
  and they get their own table.
* a **regime flip** -- the categorical call changed, whether or not the
  underlying number moved much.
* everything else, summarised by how far it moved relative to its own size.

Since I_V is an infimum over the predictor family, a *better* family should
raise estimates. A switch that lowers them broadly is evidence the new head is
worse, not merely different, so the direction of movement is reported too.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

#: A sign flip matters only if the quantity was ever far enough from zero to
#: have a sign. The threshold is unit-dependent, and applying one floor to
#: everything mislabels: a Spearman correlation going 0.048 -> -0.008 passed a
#: 0.002-"bits" test and is plainly noise around zero, while 0.002 bits is a
#: real conditional gain. Floors are therefore keyed by what the number is.
FLOORS = {
    "bits": 0.002,          # information quantities
    "spearman": 0.10,       # rank correlations: below this there is no ordering
    "share": 0.05,          # Shapley shares, fractions of a total
    "auroc": 0.005,         # matches the policy matching tolerance
    None: 0.01,             # anything unrecognised
}


def floor_for(key: str) -> float:
    k = key.lower()
    for tag, v in FLOORS.items():
        if tag and tag in k:
            return v
    return FLOORS[None]


def load(p: Path) -> dict:
    return json.loads(Path(p).read_text())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--before", required=True)
    ap.add_argument("--after", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--label-before", default="attention+FM")
    ap.add_argument("--label-after", default="concat+FM")
    a = ap.parse_args()

    before, after = load(Path(a.before)), load(Path(a.after))
    keys = sorted(set(before) | set(after))

    rows, sign_flips, regime_flips = [], [], []
    only_before, only_after = [], []
    for k in keys:
        if k not in after:
            only_before.append(k)
            continue
        if k not in before:
            only_after.append(k)
            continue
        b, c = before[k], after[k]
        if isinstance(b, str) or isinstance(c, str):
            if b != c:
                regime_flips.append({"key": k, "before": b, "after": c})
            continue
        if isinstance(b, bool) or isinstance(c, bool):
            if bool(b) != bool(c):
                regime_flips.append({"key": k, "before": b, "after": c})
            continue
        d = c - b
        scale = max(abs(b), abs(c))
        rows.append({"key": k, "before": b, "after": c, "delta": d,
                     "rel": d / scale if scale > 1e-12 else 0.0})
        if b * c < 0:
            fl = floor_for(k)
            sign_flips.append({"key": k, "before": b, "after": c, "delta": d,
                               "floor": fl,
                               "material": bool(max(abs(b), abs(c)) > fl)})

    numeric = [r for r in rows if abs(r["before"]) > 1e-9 or abs(r["after"]) > 1e-9]
    moved_up = sum(1 for r in numeric if r["delta"] > 0)
    rel = sorted((abs(r["rel"]) for r in numeric))
    med = rel[len(rel) // 2] if rel else 0.0
    biggest = sorted(numeric, key=lambda r: -abs(r["rel"]))[:25]

    out = {
        "labels": {"before": a.label_before, "after": a.label_after},
        "n_keys_compared": len(rows) + len(regime_flips),
        "n_only_before": len(only_before),
        "n_only_after": len(only_after),
        "only_before": only_before[:50],
        "only_after": only_after[:50],
        "median_abs_relative_change": med,
        "fraction_increased": moved_up / len(numeric) if numeric else 0.0,
        "n_sign_flips": len(sign_flips),
        "n_sign_flips_material": sum(1 for f in sign_flips if f["material"]),
        "sign_flips": sorted(sign_flips, key=lambda f: -abs(f["delta"])),
        "n_regime_flips": len(regime_flips),
        "regime_flips": regime_flips,
        "largest_relative_moves": biggest,
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1, sort_keys=False))

    print(f"compared {len(rows)} numeric keys")
    print(f"  median |relative change| : {med:.1%}")
    print(f"  increased                : {out['fraction_increased']:.1%} "
          f"(I_V is an infimum, so a better family should raise estimates)")
    print(f"  sign flips               : {len(sign_flips)} "
          f"({out['n_sign_flips_material']} above the floor for their unit)")
    print(f"  regime/categorical flips : {len(regime_flips)}")
    if out["n_only_before"] or out["n_only_after"]:
        print(f"  keys only before/after   : {out['n_only_before']}/{out['n_only_after']}")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
