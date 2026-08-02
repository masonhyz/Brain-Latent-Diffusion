#!/usr/bin/env python
"""Aggregate k-fold cross-validation metrics for the 7TCDM-3D pipeline.

``scripts/train_7tcdm3d.sh <name>`` writes one directory per fold,
``runs/<name>_fold{0..6}/``. Each fold logs *full held-out-set* metrics every
epoch to ``metrics_stage{1,2}.csv``; the row with the minimum ``val_loss`` is the
epoch saved as ``stage{1,2}_best.pt``, so that row is the fold's contribution to
the cross-validated estimate.

This script collects those best-epoch rows across folds and reports mean ± std —
the number you could not see with a single holdout split, because the std *is*
the split-to-split variance.

Usage:
    python scripts/aggregate_kfold.py runs/my_experiment          # stage 2 (default)
    python scripts/aggregate_kfold.py runs/my_experiment --stage 1
    python scripts/aggregate_kfold.py runs/my_experiment_fold3    # inspect one fold
    python scripts/aggregate_kfold.py runs/my_experiment --select psnr
"""
import argparse
import csv
import glob
import json
import os
import re
from pathlib import Path

import numpy as np

# Metrics logged per epoch, by stage. Stage 1 (AE) logs only losses; stage 2
# logs the full generation metrics (all brain-masked, per moyamoya/metrics.py).
STAGE_METRICS = {
    1: ["val_loss"],
    2: ["val_loss", "mae", "mse", "psnr", "ssim"],
}
HIGHER_BETTER = {"psnr", "ssim"}
LABELS = {"val_loss": "val_loss", "mae": "MAE", "mse": "MSE",
          "psnr": "PSNR", "ssim": "SSIM"}


def fold_index(d: Path):
    """Fold number from a dir named ``fold3`` (nested) or ``exp_fold3`` (legacy)."""
    m = re.search(r"fold(\d+)$", d.name)
    return int(m.group(1)) if m else -1


def find_fold_dirs(base: str):
    """Resolve a base run name to its per-fold directories, sorted by fold index.

    Three accepted forms:
      * nested parent — ``runs/exp_stage2_<ts>_cv`` containing ``fold0/ .. foldN/``
        (current layout: one timestamped parent per k-fold stage-2 run);
      * legacy flat   — ``runs/exp`` globbing sibling ``exp_fold*`` dirs;
      * a single fold — ``.../fold3`` or ``runs/exp_fold3`` (that one fold only).
    """
    base = str(base).rstrip("/")
    if re.search(r"fold\d+$", os.path.basename(base)):          # a single fold dir
        dirs = [base] if os.path.isdir(base) else []
    else:
        nested = glob.glob(os.path.join(base, "fold*"))         # current nested layout
        dirs   = nested if nested else glob.glob(f"{base}_fold*")  # else legacy flat
    dirs = [d for d in dirs if os.path.isdir(d)]
    return sorted((Path(d) for d in dirs), key=fold_index)


def read_metrics_csv(path: Path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k in list(r.keys()):
            r[k] = float(r[k])
    return rows


def select_row(rows, select: str):
    """The fold's reported epoch: best ``select`` metric (min, or max if it is a
    higher-is-better metric). Default select=val_loss matches ``*_best.pt``."""
    if select in HIGHER_BETTER:
        return max(rows, key=lambda r: r[select])
    return min(rows, key=lambda r: r[select])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("base", help="Base run name, e.g. runs/my_experiment "
                                 "(aggregates runs/my_experiment_fold*/)")
    ap.add_argument("--stage", type=int, default=2, choices=[1, 2],
                    help="Which stage's metrics to aggregate (default: 2)")
    ap.add_argument("--select", default="val_loss",
                    help="Per-fold model selection metric (default: val_loss, "
                         "matching stage{N}_best.pt). E.g. 'psnr' selects each "
                         "fold's peak-PSNR epoch.")
    ap.add_argument("--json", default=None,
                    help="Optional path to also write the summary as JSON.")
    args = ap.parse_args()

    fold_dirs = find_fold_dirs(args.base)
    if not fold_dirs:
        raise SystemExit(f"No fold directories found for '{args.base}' "
                         f"(looked for '{args.base}_fold*/').")

    metrics = STAGE_METRICS[args.stage]
    if args.select not in metrics:
        raise SystemExit(f"--select '{args.select}' is not a stage-{args.stage} "
                         f"metric; choose from {metrics}.")
    csv_name = f"metrics_stage{args.stage}.csv"

    # ── collect each fold's selected-epoch row ────────────────────────────────
    per_fold = []   # (fold_idx, epoch, {metric: value})
    missing = []
    for d in fold_dirs:
        path = d / csv_name
        if not path.exists():
            missing.append(d.name)
            continue
        rows = read_metrics_csv(path)
        if not rows:
            missing.append(d.name)
            continue
        row = select_row(rows, args.select)
        per_fold.append((fold_index(d), int(row["epoch"]),
                         {m: row[m] for m in metrics}))

    if not per_fold:
        raise SystemExit(f"Found fold dirs but no readable '{csv_name}' in any "
                         f"of them: {[d.name for d in fold_dirs]}")

    per_fold.sort(key=lambda t: t[0])

    # ── per-fold table ────────────────────────────────────────────────────────
    print(f"\nk-fold CV  |  stage {args.stage}  |  {len(per_fold)} fold(s)  "
          f"|  best epoch by {'max' if args.select in HIGHER_BETTER else 'min'} "
          f"{args.select}\n")
    header = f"{'fold':>4}  {'epoch':>6}  " + "  ".join(f"{LABELS[m]:>10}" for m in metrics)
    print(header)
    print("-" * len(header))
    for fi, ep, mv in per_fold:
        cells = "  ".join(f"{mv[m]:>10.4f}" for m in metrics)
        print(f"{fi:>4}  {ep:>6}  {cells}")

    # ── aggregate: mean ± std across folds ────────────────────────────────────
    print("-" * len(header))
    summary = {}
    mean_cells, std_cells = [], []
    for m in metrics:
        vals = np.array([mv[m] for _, _, mv in per_fold], dtype=float)
        mean, std = float(vals.mean()), float(vals.std(ddof=0))
        summary[m] = {"mean": mean, "std": std,
                      "per_fold": [float(mv[m]) for _, _, mv in per_fold]}
        mean_cells.append(f"{mean:>10.4f}")
        std_cells.append(f"{std:>10.4f}")
    print(f"{'mean':>4}  {'':>6}  " + "  ".join(mean_cells))
    print(f"{'std':>4}  {'':>6}  " + "  ".join(std_cells))

    print("\nCross-validated (mean ± std across folds):")
    for m in metrics:
        print(f"  {LABELS[m]:<8}: {summary[m]['mean']:.4f} ± {summary[m]['std']:.4f}")

    if missing:
        print(f"\n[warn] {len(missing)} fold dir(s) had no usable {csv_name}: "
              f"{', '.join(missing)}")

    if args.json:
        out = {"base": args.base, "stage": args.stage, "select": args.select,
               "n_folds_found": len(per_fold),
               "folds": [{"fold": fi, "epoch": ep, **mv} for fi, ep, mv in per_fold],
               "summary": summary, "missing": missing}
        with open(args.json, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nSummary JSON → {args.json}")


if __name__ == "__main__":
    main()
