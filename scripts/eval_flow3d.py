"""
Flow3D evaluation — brain-masked metrics, always reported against the identity
baseline (predict x_post := x_pre).

    # validation split of the run's own recorded split
    python scripts/eval_flow3d.py --ckpt runs/flow3d/best_mae.pt --val_only

    # full dataset, no PNGs
    python scripts/eval_flow3d.py --ckpt runs/flow3d/best_mae.pt --no-grids

    # how few ODE steps can we get away with? (the flow-matching selling point)
    python scripts/eval_flow3d.py --ckpt runs/flow3d/best_mae.pt --val_only \
        --sweep_steps 1 2 4 8 16 32

    # one subject, ensemble of 5 perturbed trajectories, export NIfTI
    python scripts/eval_flow3d.py --ckpt runs/flow3d/best_mae.pt \
        --subject 2024_040 --n_samples 5 --init_noise 0.05 --save_nii

Outputs (under --out_dir):
    grids/<id>.png     per-sample orthogonal grids incl. an error column
    metrics.csv        per-sample model *and* identity-baseline metrics
    summary.json       run config + aggregates + the model-vs-identity verdict
"""

import argparse
import csv
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from moyamoya.data import reconstruct_val_split
from moyamoya.dataset import PrePostFMRI
from moyamoya.metrics import compute_metrics, tissue_mask
from moyamoya.models.flow3d import load_flow3d_checkpoint
from moyamoya.runlog import save_grid
from moyamoya.transform import ToChannelsFirstAndNormalize
from moyamoya.utils import get_device
from moyamoya.viz import percentile_norm as _to_np

METRICS = ("mae", "mse", "psnr", "ssim")
HIGHER_BETTER = {"mae": False, "mse": False, "psnr": True, "ssim": True}


def subject_seed(base: int, sid: str) -> int:
    """Stable per-subject RNG seed, so a subject's prediction reproduces
    regardless of dataset ordering or batch-vs-single mode."""
    h = int(hashlib.sha1(sid.encode()).hexdigest()[:8], 16)
    return (base + h) % (2 ** 31 - 1)


def save_nifti(pred_b: torch.Tensor, pre_path: str, out_path: Path):
    """Write the prediction back to NIfTI in the source geometry.

    The transform did (X,Y,Z) → permute(2,1,0) → (Z,Y,X); invert that and reuse
    the source pre-surgery affine/header so the volume overlays the original
    scan. Intensities stay in z-score-normalised units.
    """
    import nibabel as nib

    pred_zyx = pred_b.squeeze().cpu().float().numpy()
    pred_xyz = np.ascontiguousarray(np.transpose(pred_zyx, (2, 1, 0))).astype(np.float32)
    pre_img = nib.load(pre_path)
    if pred_xyz.shape != pre_img.shape:
        raise SystemExit(f"shape mismatch {pred_xyz.shape} vs source {pre_img.shape}")
    hdr = pre_img.header.copy()
    hdr.set_data_dtype(np.float32)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(pred_xyz, affine=pre_img.affine, header=hdr), out_path)
    return pred_xyz.shape


def aggregate(vals) -> dict:
    a = np.asarray(vals, dtype=float)
    return {"mean": float(a.mean()), "std": float(a.std()),
            "median": float(np.median(a)), "min": float(a.min()), "max": float(a.max())}


@torch.no_grad()
def predict(model, x, args, steps: int, seed: int):
    """Sample the model, averaging ``--n_samples`` trajectories.

    Averaging only does something when ``--init_noise > 0``: the probability-flow
    ODE is deterministic given its start, so with an unperturbed start every draw
    is identical.
    """
    g = torch.Generator(device=x.device).manual_seed(seed)
    preds = [model.sample(x, steps=steps, solver=args.solver,
                          guidance_scale=args.guidance_scale,
                          init_noise=args.init_noise, generator=g)
             for _ in range(args.n_samples)]
    return torch.stack(preds).mean(0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root",   type=str, default="fmri")
    p.add_argument("--ckpt",        type=str, default="runs/flow3d/best_mae.pt")
    p.add_argument("--out_dir",     type=str, default=None,
                   help="Default: outputs/predict/<subject> for --subject, "
                        "else outputs/eval/<run name>")
    p.add_argument("--pre_dirname",  type=str, default="pre_surgery")
    p.add_argument("--post_dirname", type=str, default="6_months_post_surgery")
    p.add_argument("--subject",   type=str, default=None,
                   help="Evaluate a single subject id (e.g. 2024_040)")
    p.add_argument("--val_only",  action="store_true",
                   help="Restrict to the checkpoint's held-out validation split")
    p.add_argument("--val_frac",  type=float, default=None,
                   help="Override the split fraction recorded in the checkpoint")
    p.add_argument("--seed",      type=int, default=None,
                   help="Override the split seed recorded in the checkpoint")
    # sampling
    p.add_argument("--steps",  type=int, default=None,
                   help="ODE steps (default: the value the run was configured with)")
    p.add_argument("--solver", type=str, default=None, choices=["euler", "heun", "rk4"])
    p.add_argument("--sweep_steps", type=int, nargs="+", default=None,
                   help="Evaluate at each of these step counts and print an "
                        "accuracy-vs-NFE table instead of a single run")
    p.add_argument("--guidance_scale", type=float, default=None)
    p.add_argument("--init_noise",     type=float, default=0.0,
                   help="Std of noise on x(0); required for --n_samples to vary")
    p.add_argument("--n_samples",  type=int, default=1,
                   help="Average this many trajectories per subject")
    p.add_argument("--sample_seed", type=int, default=0)
    p.add_argument("--use_ema",    dest="use_ema", action="store_true")
    p.add_argument("--no-ema",     dest="use_ema", action="store_false")
    p.set_defaults(use_ema=True)
    # outputs
    p.add_argument("--save_grids", dest="save_grids", action="store_true")
    p.add_argument("--no-grids",   dest="save_grids", action="store_false")
    p.set_defaults(save_grids=True)
    p.add_argument("--save_nii", action="store_true",
                   help="Write the predicted volume to .nii.gz in source geometry")
    args = p.parse_args()

    device = get_device()
    model, raw = load_flow3d_checkpoint(args.ckpt, device=device, use_ema=args.use_ema)
    ck = raw.get("args", {})

    steps = args.steps or ck.get("steps", 8)
    args.solver = args.solver or ck.get("solver", "heun")
    if args.guidance_scale is None:
        args.guidance_scale = ck.get("guidance_scale", 1.0)

    print(f"Checkpoint: {args.ckpt}  ({'EMA' if args.use_ema else 'online'} weights)")
    print(f"  source={ck.get('source')} sigma={ck.get('sigma')} "
          f"trained {ck.get('epochs')} epochs, seed {ck.get('seed')}")
    print(f"Device: {device}  |  solver={args.solver} steps={steps} "
          f"guidance={args.guidance_scale} init_noise={args.init_noise} "
          f"n_samples={args.n_samples}")

    # ── dataset / split ──────────────────────────────────────────────────────
    # Preprocessing must match what the checkpoint was trained with, or the
    # background alone shifts every metric by ~2x on MAE.
    zero_bg = ck.get("zero_background", False)
    print(f"  preprocessing: zero_background={zero_bg} "
          f"({'brain-only' if zero_bg else 'whole-volume'} metrics)")
    ds = PrePostFMRI(root_dir=args.data_root, pre_dirname=args.pre_dirname,
                     post_dirname=args.post_dirname,
                     transform=ToChannelsFirstAndNormalize(nonzero_mask=True,
                                                           zero_background=zero_bg),
                     strict=False, return_paths=True)

    if args.subject:
        ids = [n.replace(".nii.gz", "") for n, _, _ in ds.pairs]
        if args.subject not in ids:
            raise SystemExit(f"subject {args.subject!r} not found (have {len(ids)} subjects)")
        indices = [ids.index(args.subject)]
        scope = f"subject {args.subject}"
    elif args.val_only:
        seed = args.seed if args.seed is not None else ck.get("seed", 42)
        vf = args.val_frac if args.val_frac is not None else ck.get("val_frac", 0.15)
        _, val = reconstruct_val_split(ds, vf, seed,
                                       n_folds=ck.get("n_folds"), fold=ck.get("fold"))
        indices = list(val.indices)
        scope = (f"validation split (seed={seed}, "
                 + (f"fold {ck.get('fold')}/{ck.get('n_folds')}"
                    if ck.get("fold") is not None else f"val_frac={vf}") + ")")
    else:
        indices = list(range(len(ds)))
        scope = "full dataset"
    print(f"Scope: {scope} — {len(indices)} subjects\n")

    out_dir = Path(args.out_dir or (f"outputs/predict/{args.subject}" if args.subject
                                    else f"outputs/eval/{Path(args.ckpt).parent.name}"))
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── optional step sweep: accuracy vs NFE ─────────────────────────────────
    if args.sweep_steps:
        nfe_mult = {"euler": 1, "heun": 2, "rk4": 4}[args.solver]
        print(f"{'steps':>6} {'NFE':>5} " + " ".join(f"{m.upper():>8}" for m in METRICS))
        sweep = {}
        for s in args.sweep_steps:
            acc = {m: [] for m in METRICS}
            for i in indices:
                x, y, meta = ds[i]
                xb = x.unsqueeze(0).to(device)
                pred = predict(model, xb, args, s, subject_seed(args.sample_seed, meta["id"]))
                mm = compute_metrics(pred[0].float().cpu(), y, tissue_mask(x, y))
                for m in METRICS:
                    acc[m].append(mm[m])
            means = {m: float(np.mean(v)) for m, v in acc.items()}
            sweep[s] = means
            print(f"{s:>6} {s * nfe_mult:>5} " + " ".join(f"{means[m]:8.4f}" for m in METRICS))
        with open(out_dir / "step_sweep.json", "w") as f:
            json.dump({"solver": args.solver, "scope": scope, "sweep": sweep}, f, indent=2)
        print(f"\nSweep saved: {out_dir / 'step_sweep.json'}")
        return

    # ── main evaluation ──────────────────────────────────────────────────────
    rows = []
    model_acc = {m: [] for m in METRICS}
    ident_acc = {m: [] for m in METRICS}

    for n, i in enumerate(indices, 1):
        x, y, meta = ds[i]
        sid = meta["id"]
        xb = x.unsqueeze(0).to(device)

        pred = predict(model, xb, args, steps, subject_seed(args.sample_seed, sid))
        pred_c = pred[0].float().cpu()

        mask = tissue_mask(x, y)
        mm = compute_metrics(pred_c, y, mask)
        # The same volume scored against the trivial predictor, subject by
        # subject — so per-subject wins and losses are both visible, not just
        # the aggregate.
        bm = compute_metrics(x, y, mask)

        for m in METRICS:
            model_acc[m].append(mm[m])
            ident_acc[m].append(bm[m])
        rows.append({"id": sid, **{m: mm[m] for m in METRICS},
                     **{f"identity_{m}": bm[m] for m in METRICS}})

        if args.save_grids:
            save_grid(_to_np(xb), _to_np(y.unsqueeze(0)), _to_np(pred),
                      f"{sid}  |  MAE={mm['mae']:.4f} (identity {bm['mae']:.4f})  "
                      f"PSNR={mm['psnr']:.2f}  SSIM={mm['ssim']:.4f}",
                      out_dir / "grids" / f"{sid}.png")
        if args.save_nii:
            shape = save_nifti(pred, meta["pre_path"], out_dir / f"{sid}_pred.nii.gz")
            print(f"  wrote {out_dir / f'{sid}_pred.nii.gz'} {shape}")

        if n % 10 == 0 or n == len(indices):
            print(f"  [{n}/{len(indices)}] running MAE "
                  f"model={np.mean(model_acc['mae']):.4f} "
                  f"identity={np.mean(ident_acc['mae']):.4f}")

    # ── per-sample CSV ───────────────────────────────────────────────────────
    csv_path = out_dir / "metrics.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nPer-sample metrics: {csv_path}")

    # ── verdict ──────────────────────────────────────────────────────────────
    print(f"\n=== {scope} — n={len(indices)} ===")
    print(f"{'metric':>7}  {'Flow3D':>18}  {'identity':>9}  {'Δ':>9}  verdict")
    verdict = {}
    for m in METRICS:
        mu, sd = float(np.mean(model_acc[m])), float(np.std(model_acc[m]))
        bl = float(np.mean(ident_acc[m]))
        beat = (mu > bl) if HIGHER_BETTER[m] else (mu < bl)
        delta = mu - bl
        verdict[m] = bool(beat)
        print(f"{m.upper():>7}  {mu:9.4f} ± {sd:6.4f}  {bl:9.4f}  {delta:+9.4f}  "
              f"{'BEATS identity' if beat else 'loses to identity'}")

    # per-subject win rate: an aggregate can hide a model that is great on a few
    # subjects and harmful on the rest.
    wins = sum(1 for r in rows if r["mae"] < r["identity_mae"])
    print(f"\nPer-subject MAE wins over identity: {wins}/{len(rows)} "
          f"({100 * wins / len(rows):.0f}%)")

    summary = {
        "timestamp": datetime.now().isoformat(),
        "ckpt": args.ckpt, "use_ema": args.use_ema, "scope": scope,
        "n_subjects": len(indices),
        "sampling": {"solver": args.solver, "steps": steps,
                     "nfe": steps * {"euler": 1, "heun": 2, "rk4": 4}[args.solver],
                     "guidance_scale": args.guidance_scale,
                     "init_noise": args.init_noise, "n_samples": args.n_samples,
                     "sample_seed": args.sample_seed},
        "train_args": ck,
        "model":    {m: aggregate(model_acc[m]) for m in METRICS},
        "identity": {m: aggregate(ident_acc[m]) for m in METRICS},
        "beats_identity": verdict,
        "per_subject_mae_wins": {"wins": wins, "total": len(rows)},
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"Summary saved: {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
