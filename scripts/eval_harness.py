"""
Multi-model evaluation harness — score many models the SAME way and collect one
comparison table (whole-volume + change-region + coherent metrics).

Each model is evaluated on **its own recorded held-out split** (the seed/val_frac
stored in its checkpoint), so no model is scored on a subject it trained on. The
flip side: different models used different splits, so their subject sets differ and
the table is *roughly*, not strictly, comparable — the split is printed per model
and the harness warns when they disagree.

Currently wired for **flow3d** checkpoints (delegates to scripts/eval_flow3d.py, one
subprocess per model, so GPU memory is freed between models). Diffusion families
(LDM / CDM3D / 7TCDM) are a documented hook — see FAMILY_ADAPTERS below; each just
needs to emit a summary.json in the same schema.

Usage
-----
    # explicit checkpoints
    GPU=0 python scripts/eval_harness.py \
        --ckpts /data1/masonhu/runs/flow3d_2026-08-02_21-07-41/best_mae.pt \
                /data1/masonhu/runs/flow3d_sharp/best_mae.pt

    # auto-discover every flow3d run under a directory (best_mae.pt of each)
    GPU=0 python scripts/eval_harness.py --runs_dir /data1/masonhu/runs --select best_mae

    # see what would run, without running it
    python scripts/eval_harness.py --runs_dir /data1/masonhu/runs --dry_run

Outputs (under --out_dir, default outputs/harness/<timestamp>/):
    <label>/summary.json   the per-model eval_flow3d summary
    comparison.csv         one row per model (the paper table, tidy)
    comparison.json        the same, structured
"""

import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EVAL_FLOW3D = ROOT / "scripts" / "eval_flow3d.py"

# Family → the script that evaluates one checkpoint of that family into a
# summary.json with the shared schema. Only flow3d is wired today; to add a
# diffusion family, point it at an eval script that writes the same summary.json
# (model/identity aggregates + optional change_region block) and add detection to
# detect_family().
FAMILY_ADAPTERS = {"flow3d": EVAL_FLOW3D}


def detect_family(ckpt: Path) -> str:
    """Best-effort model family from the checkpoint, without loading the weights.

    flow3d runs carry a sibling ``hparams.json`` and flow-matching-specific train
    args (``source``/``sigma``). Everything else is reported as ``unknown`` so the
    harness can skip it with a clear message rather than mis-scoring it.
    """
    hp = ckpt.parent / "hparams.json"
    if hp.exists():
        try:
            a = json.loads(hp.read_text())
            a = a.get("args", a)
            if "source" in a and "sigma" in a:
                return "flow3d"
        except (json.JSONDecodeError, OSError):
            pass
    # name-based fallback: the run dirs here are flow3d_* by convention
    if ckpt.parent.name.startswith("flow3d"):
        return "flow3d"
    return "unknown"


def resolve_ckpts(args) -> list[tuple[str, Path]]:
    """→ list of (label, ckpt_path). From --ckpts, or globbed from --runs_dir."""
    out: list[tuple[str, Path]] = []
    if args.ckpts:
        paths = [Path(c) for c in args.ckpts]
    else:
        root = Path(args.runs_dir)
        # flow3d_* run dirs, each contributing its <select>.pt if present
        paths = sorted(p / f"{args.select}.pt" for p in root.glob("flow3d*")
                       if (p / f"{args.select}.pt").exists())
        if not paths:
            raise SystemExit(f"no flow3d*/{args.select}.pt under {root}")
    labels = args.labels or [p.parent.name for p in paths]
    if len(labels) != len(paths):
        raise SystemExit(f"--labels has {len(labels)} entries for {len(paths)} ckpts")
    for lab, p in zip(labels, paths):
        if not p.exists():
            raise SystemExit(f"checkpoint not found: {p}")
        out.append((lab, p))
    return out


def run_one(label: str, ckpt: Path, args, out_dir: Path) -> Path | None:
    """Evaluate one checkpoint → its summary.json path (or None on failure/skip)."""
    fam = detect_family(ckpt)
    if fam not in FAMILY_ADAPTERS:
        print(f"  ! {label}: family '{fam}' has no adapter — skipping. "
              f"(Add one in FAMILY_ADAPTERS; diffusion is not wired yet.)")
        return None
    model_out = out_dir / label
    summary = model_out / "summary.json"
    if args.reuse and summary.exists():
        print(f"  = {label}: reusing existing {summary}")
        return summary
    cmd = [sys.executable, str(FAMILY_ADAPTERS[fam]), "--ckpt", str(ckpt),
           "--data_root", args.data_root, "--no-grids",
           "--out_dir", str(model_out),
           "--coherent" if args.coherent else "--no-coherent"]
    if not args.full_dataset:
        cmd.append("--val_only")
    for flag in ("steps", "solver", "init_noise", "sample_seed"):
        v = getattr(args, flag)
        if v is not None:
            cmd += [f"--{flag}", str(v)]
    print(f"  → {label}: {fam}  ({' '.join(cmd[2:])})")
    if args.dry_run:
        return None
    r = subprocess.run(cmd, env=None)
    if r.returncode != 0 or not summary.exists():
        print(f"  ! {label}: eval failed (exit {r.returncode}); no summary written")
        return None
    return summary


def split_key(s: dict) -> str:
    """A compact identifier of the split a model was scored on."""
    ta = s.get("train_args", {}) or {}
    if ta.get("fold") is not None:
        return f"fold{ta['fold']}/{ta.get('n_folds')}#{ta.get('seed')}"
    return f"seed{ta.get('seed')}·vf{ta.get('val_frac')}"


def row_from_summary(label: str, s: dict) -> dict:
    """Flatten one eval_flow3d summary into a comparison row (means only)."""
    m, idn = s.get("model", {}), s.get("identity", {})
    cr = s.get("change_region", {})
    w = s.get("per_subject_mae_wins", {})
    row = {
        "model": label,
        "n": s.get("n_subjects"),
        "split": split_key(s),
        "mae": m.get("mae", {}).get("mean"),
        "identity_mae": idn.get("mae", {}).get("mean"),
        "psnr": m.get("psnr", {}).get("mean"),
        "ssim": m.get("ssim", {}).get("mean"),
        "win_rate": (w.get("wins", 0) / w["total"]) if w.get("total") else None,
        # change region (present when the eval ran with --coherent)
        "change_mae_improvement": cr.get("change_mae_improvement"),
        "coherent_frac": cr.get("coherent_frac"),
        "coherent_mae_improvement": cr.get("coherent_mae_improvement"),
        "edge_enrichment": cr.get("edge_enrichment"),
    }
    return row


def _fmt(v, nd=4):
    return f"{v:.{nd}f}" if isinstance(v, (int, float)) else "  —  "


def print_table(rows: list[dict], splits_agree: bool):
    """Ranked table: best coherent improvement first (falls back to whole-vol MAE)."""
    def sort_key(r):
        ci = r.get("coherent_mae_improvement")
        return (-ci if isinstance(ci, (int, float)) else float("inf"),
                r.get("mae") if isinstance(r.get("mae"), (int, float)) else float("inf"))
    rows = sorted(rows, key=sort_key)
    cols = [("model", 26, "s"), ("n", 4, "d"), ("mae", 8, "f"), ("identity_mae", 9, "f"),
            ("win_rate", 6, "p"), ("change_mae_improvement", 9, "f"),
            ("coherent_mae_improvement", 9, "f"), ("coherent_frac", 6, "f")]
    hdr = {"model": "model", "n": "n", "mae": "MAE↓", "identity_mae": "ident",
           "win_rate": "win%", "change_mae_improvement": "rawΔ↑",
           "coherent_mae_improvement": "cohΔ↑", "coherent_frac": "cohFr"}
    line = "  ".join(f"{hdr[k]:>{w}}" if k != "model" else f"{hdr[k]:<{w}}"
                     for k, w, _ in cols)
    print("\n" + line)
    print("  ".join("-" * w for _, w, _ in cols))
    for r in rows:
        cells = []
        for k, w, kind in cols:
            v = r.get(k)
            if kind == "s":
                cells.append(f"{str(v)[:w]:<{w}}")
            elif kind == "d":
                cells.append(f"{v:>{w}d}" if isinstance(v, int) else f"{'—':>{w}}")
            elif kind == "p":
                cells.append(f"{100*v:>{w-1}.0f}%" if isinstance(v, float) else f"{'—':>{w}}")
            else:
                cells.append(f"{_fmt(v):>{w}}")
        print("  ".join(cells))
    print("\n(cohΔ↑ = coherent_mae_improvement — the honest change headline; "
          "rawΔ↑ inflated by the ~60% registration-noise ROI.)")
    if not splits_agree:
        print("\n  ⚠ models were scored on DIFFERENT held-out splits (see 'split' "
              "column) — subject sets differ, so the table is approximate, not\n"
              "    a strict head-to-head. Use a common global test set for the paper.")


def main():
    p = argparse.ArgumentParser(description="Evaluate many models the same way.")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--ckpts", nargs="+", help="Explicit checkpoint paths")
    src.add_argument("--runs_dir", type=str, help="Glob flow3d*/<select>.pt under here")
    p.add_argument("--select", default="best_mae",
                   help="Which checkpoint per run dir (best_mae|best|last|best_mse|...)")
    p.add_argument("--labels", nargs="+", help="Names for the models (default: run dir)")
    p.add_argument("--data_root", default="/data1/masonhu/fmri")
    p.add_argument("--out_dir", default=None)
    p.add_argument("--coherent", dest="coherent", action="store_true", default=True,
                   help="Report change-region + coherent metrics (default on)")
    p.add_argument("--no-coherent", dest="coherent", action="store_false")
    p.add_argument("--full_dataset", action="store_true",
                   help="Score the whole dataset instead of each model's val split "
                        "(WARNING: includes training subjects — leakage)")
    p.add_argument("--reuse", action="store_true",
                   help="Skip a model whose summary.json already exists in --out_dir")
    p.add_argument("--dry_run", action="store_true",
                   help="List what would be evaluated, then stop")
    # sampling passthrough (default: each checkpoint's recorded values)
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--solver", type=str, default=None, choices=["euler", "heun", "rk4"])
    p.add_argument("--init_noise", type=float, default=None)
    p.add_argument("--sample_seed", type=int, default=None)
    args = p.parse_args()

    ckpts = resolve_ckpts(args)
    out_dir = Path(args.out_dir or f"outputs/harness/{datetime.now():%Y-%m-%d_%H-%M-%S}")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[harness] {len(ckpts)} model(s) → {out_dir}"
          + ("   (dry run)" if args.dry_run else ""))

    summaries = []
    for label, ckpt in ckpts:
        sp = run_one(label, ckpt, args, out_dir)
        if sp is not None:
            summaries.append((label, json.loads(Path(sp).read_text())))
    if args.dry_run:
        return
    if not summaries:
        raise SystemExit("no models evaluated successfully")

    rows = [row_from_summary(lab, s) for lab, s in summaries]
    splits = {r["split"] for r in rows}
    print_table(rows, splits_agree=(len(splits) == 1))

    cols = list(rows[0].keys())
    with open(out_dir / "comparison.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    (out_dir / "comparison.json").write_text(json.dumps(rows, indent=2, default=str))
    print(f"\nComparison table: {out_dir / 'comparison.csv'}")


if __name__ == "__main__":
    main()
