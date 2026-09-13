#!/usr/bin/env python3
"""Reproduce every number and figure in the paper, end to end.

    python scripts/run_all.py                # full run (~1.5 h on 4 CPU cores)
    python scripts/run_all.py --quick        # smoke run (~5 min)

Stages, in the order the paper presents them:

1. ``theory``      -- numerically certify Lemmas 1-3 and Theorems 1-3
2. ``cohort``      -- build the simulated MIMIC-shaped cohort (or use a real one)
3. ``simulation``  -- validate the estimator against exact ground truth
4. ``analysis``    -- the main analysis, one run per endpoint
5. ``sensitivity`` -- repeat under missing-not-at-random test ordering
6. ``paper``       -- regenerate the manuscript's numbers and tables

Pass ``--mimic-root`` to run stages 4-5 on a real PhysioNet extraction instead of
the simulator; the analysis code is identical either way.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def run(cmd: list[str], log_path: Path) -> float:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"\n$ {' '.join(cmd)}", flush=True)
    t0 = time.time()
    # Python block-buffers stdout when it is a file, so a multi-hour stage would
    # show no progress at all until it finished -- which reads exactly like a
    # hang. Unbuffered output costs nothing here and makes the run observable.
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    with open(log_path, "w") as fh:
        proc = subprocess.run(cmd, cwd=REPO, stdout=fh, stderr=subprocess.STDOUT,
                              env=env)
    dt = time.time() - t0
    if proc.returncode != 0:
        print(f"  FAILED after {dt:.0f}s -- see {log_path}", flush=True)
        print(log_path.read_text()[-3000:], flush=True)
        raise SystemExit(proc.returncode)
    print(f"  ok in {dt:.0f}s  (log: {log_path})", flush=True)
    return dt


def find_stale_runs() -> list[str]:
    """Other INFOGAIN stage processes already running on this machine.

    Two suites sharing four cores do not take turns; they each take twice as
    long, and if both reach the same output directory the second silently
    overwrites the first. A leftover stage from an interrupted run is easy to
    miss -- ``pkill -f run_all.py`` matches the shell that issues it and kills
    that instead, leaving the stage orphaned under init -- so check rather than
    assume.
    """
    try:
        out = subprocess.run(["pgrep", "-af", "infogain.experiments.run_"],
                             capture_output=True, text=True, timeout=10).stdout
    except (FileNotFoundError, subprocess.SubprocessError):
        return []
    me = str(os.getpid())
    rows = []
    for line in out.splitlines():
        pid, _, args = line.partition(" ")
        if pid in (me, str(os.getppid())) or "pgrep" in args:
            continue
        rows.append(f"  pid {pid}: {args[:110]}")
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--n", type=int, default=50000, help="simulated cohort size")
    ap.add_argument("--epochs", type=int, default=130)
    ap.add_argument("--folds", type=int, default=4)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--results", default="results")
    ap.add_argument("--mimic-root", default=None,
                    help="run the analysis on a real PhysioNet extraction")
    ap.add_argument("--outcomes", default="mortality_30d,hf_readmission_30d,aki_7d,icu_transfer_48h")
    ap.add_argument("--skip", default="", help="comma-separated stage names to skip")
    ap.add_argument("--force", action="store_true",
                    help="start even if other INFOGAIN stages are already running")
    args = ap.parse_args()

    stale = find_stale_runs()
    if stale and not args.force:
        print("Refusing to start: INFOGAIN stages are already running.\n"
              + "\n".join(stale)
              + "\n\nKill them by PID (not `pkill -f`, which matches the shell "
                "issuing it), or pass --force.")
        raise SystemExit(2)

    py = sys.executable
    res = Path(args.results)
    logs = res / "logs"
    skip = {s.strip() for s in args.skip.split(",") if s.strip()}
    outcomes = [o.strip() for o in args.outcomes.split(",") if o.strip()]
    timings: dict[str, float] = {}

    if args.quick:
        args.n, args.epochs, args.folds, args.seeds = 6000, 25, 3, 1

    # -- 1. theory ---------------------------------------------------------- #
    if "theory" not in skip:
        cmd = [py, "-m", "infogain.theory.verify", "--out", str(res / "theory" / "verification.json")]
        if args.quick:
            cmd.append("--quick")
        timings["theory"] = run(cmd, logs / "theory.log")

    # -- 2. cohort ---------------------------------------------------------- #
    if args.mimic_root:
        cohort_dir = Path("data/cohort_mimic")
        if "cohort" not in skip:
            timings["cohort"] = run(
                [py, "-m", "infogain.data.mimic_cohort", "--root", args.mimic_root,
                 "--out", str(cohort_dir), "--split", "temporal"],
                logs / "cohort.log")
    else:
        cohort_dir = Path("data/cohort_synthetic")
        if "cohort" not in skip:
            timings["cohort"] = run(
                [py, "scripts/make_synthetic_cohort.py", "--n", str(args.n),
                 "--out", str(cohort_dir)], logs / "cohort.log")

    # -- 3. simulation validation ------------------------------------------- #
    if "simulation" not in skip and not args.mimic_root:
        cmd = [py, "-m", "infogain.experiments.run_simulation",
               "--out", str(res / "simulation"), "--epochs", str(args.epochs),
               "--folds", str(args.folds), "--seeds", str(args.seeds)]
        if args.quick:
            cmd.append("--quick")
        timings["simulation"] = run(cmd, logs / "simulation.log")

    # -- 4. main analysis, per endpoint ------------------------------------- #
    if "analysis" not in skip:
        for oc in outcomes:
            cmd = [py, "-m", "infogain.experiments.run_cohort",
                   "--cohort", str(cohort_dir), "--outcome", oc,
                   "--out", str(res / cohort_dir.name / oc),
                   "--epochs", str(args.epochs), "--folds", str(args.folds),
                   "--seeds", str(args.seeds)]
            if args.quick:
                cmd.append("--quick")
            timings[f"analysis:{oc}"] = run(cmd, logs / f"analysis_{oc}.log")

    # -- 5. MNAR sensitivity ------------------------------------------------- #
    if "sensitivity" not in skip and not args.mimic_root:
        mnar_dir = Path("data/cohort_synthetic_mnar")
        timings["sensitivity_cohort"] = run(
            [py, "scripts/make_synthetic_cohort.py", "--n", str(args.n),
             "--mnar", "1.2", "--out", str(mnar_dir)], logs / "cohort_mnar.log")
        cmd = [py, "-m", "infogain.experiments.run_cohort",
               "--cohort", str(mnar_dir), "--outcome", outcomes[0],
               "--out", str(res / "sensitivity_mnar" / outcomes[0]),
               "--epochs", str(args.epochs), "--folds", str(args.folds),
               "--seeds", str(args.seeds), "--no-figures"]
        if args.quick:
            cmd.append("--quick")
        timings["sensitivity"] = run(cmd, logs / "sensitivity.log")

    # -- 6. refresh the manuscript's numbers and tables ---------------------- #
    if "paper" not in skip:
        cohort_key = cohort_dir.name
        timings["paper_numbers"] = run(
            [py, "scripts/make_paper_numbers.py", "--results", str(res),
             "--cohort-name", cohort_key, "--primary", outcomes[0]],
            logs / "paper_numbers.log")
        timings["paper_tables"] = run(
            [py, "scripts/make_paper_tables.py", "--results", str(res),
             "--cohort-name", cohort_key, "--primary", outcomes[0]],
            logs / "paper_tables.log")

    (res).mkdir(parents=True, exist_ok=True)
    with open(res / "run_manifest.json", "w") as fh:
        json.dump({"args": vars(args), "timings_seconds": timings,
                   "total_seconds": sum(timings.values())}, fh, indent=2)
    print(f"\nall stages complete in {sum(timings.values()) / 60:.1f} min")
    print(f"manifest -> {res / 'run_manifest.json'}")
    print("build the manuscript with:  bash scripts/build_paper.sh")


if __name__ == "__main__":
    main()
