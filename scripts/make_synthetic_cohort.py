#!/usr/bin/env python3
"""Generate and save a simulated MIMIC-shaped cohort with its exact ground truth."""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

from infogain.data.synthetic import default_design, generate, ground_truth_table
from infogain.utils.io import save_json


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=50000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mnar", type=float, default=0.0,
                    help="strength of missing-not-at-random ordering")
    ap.add_argument("--out", default="data/cohort_synthetic")
    ap.add_argument("--save-ground-truth", action="store_true")
    args = ap.parse_args()

    design = default_design()
    design.mnar_strength = args.mnar
    cohort, gt = generate(design, n=args.n, seed=args.seed)
    out = Path(args.out)
    cohort.save(out)
    print(cohort.describe())

    tables = {}
    for oc in cohort.outcome_names:
        df = ground_truth_table(gt, oc, baseline=sorted(cohort.spec.baseline),
                                respect_observation=True)
        df.to_csv(out / f"ground_truth_{oc}.csv", index=False)
        tables[oc] = df.to_dict("records")
        print(f"\n== exact decomposition: {oc} ==")
        print(df.round(5).to_string(index=False))
    save_json({"n": args.n, "seed": args.seed, "mnar": args.mnar,
               "ground_truth": tables}, out / "ground_truth.json")

    if args.save_ground_truth:
        with open(out / "ground_truth.pkl", "wb") as fh:
            pickle.dump(gt, fh)
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
