#!/usr/bin/env python3
"""Reproduce the variance-reduction factor quoted in ``vinfo.core.paired_gain``.

That docstring used to claim pairing buys "roughly an order of magnitude" in
standard error.  It does not: the factor is
``sqrt(var(a) + var(b)) / sd(a - b)``, set by how strongly the two PVI vectors
correlate, and on fitted models that is 1.8x to 5.9x rather than 10x.  This
script is what produced those numbers, kept so the docstring stays checkable
rather than becoming folklore again.

The factor does not depend on the cohort size -- both standard errors scale as
1/sqrt(n) -- so a small fit measures it faithfully.

    python3 scripts/check_paired_se.py [--n 6000] [--epochs 30]
"""
from __future__ import annotations

import argparse
import logging
import warnings

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=6000)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--folds", type=int, default=3)
    ap.add_argument("--outcome", default="mortality_30d")
    args = ap.parse_args()

    warnings.filterwarnings("ignore")
    logging.disable(logging.INFO)
    from infogain.data.synthetic import generate
    from infogain.encoders.train import TrainConfig, fit_family

    cohort, _ = generate(n=args.n, seed=0)
    cf = fit_family(cohort, args.outcome,
                    TrainConfig(epochs=args.epochs, n_folds=args.folds,
                                seeds=(0,), patience=8), keep_models=True)
    full = frozenset(cohort.spec.names)

    print(f"{'modality':10s} {'sd(a-b)':>10s} {'sqrt(va+vb)':>12s} "
          f"{'ratio':>7s} {'corr':>8s}")
    ratios = []
    for m in cohort.spec.orderable:
        a, b = cf.pvi(full), cf.pvi(full - {m})
        paired = float(np.std(a - b, ddof=1))
        unpaired = float(np.sqrt(np.var(a, ddof=1) + np.var(b, ddof=1)))
        ratio = unpaired / max(paired, 1e-12)
        ratios.append(ratio)
        print(f"{m:10s} {paired:10.4f} {unpaired:12.4f} {ratio:6.1f}x "
              f"{np.corrcoef(a, b)[0, 1]:8.4f}")
    print(f"\nrange {min(ratios):.1f}x - {max(ratios):.1f}x   "
          f"median {float(np.median(ratios)):.1f}x")


if __name__ == "__main__":
    main()
