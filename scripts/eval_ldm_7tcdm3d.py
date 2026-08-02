"""
eval_ldm_7tcdm3d.py

Evaluate / predict with a 7TCDM-3D Latent Diffusion stage-2 checkpoint.

This is the single entry point for both batch evaluation and single-subject
prediction (it absorbs the former predict_ldm_7tcdm3d_nii.py). For every
evaluated subject it optionally renders a 3×3 grid

    rows : Axial | Coronal | Sagittal
    cols : Pre-surgery | Prediction | Post-surgery GT

and can export the predicted volume as NIfTI in the source geometry.

Reproducibility: DDIM sampling is seeded per subject from --sample_seed, so a
subject's prediction is identical whether produced in a full-dataset run or a
single-subject run, and reruns match bit-for-bit.

Modes:
    # full dataset
    python scripts/eval_ldm_7tcdm3d.py --ckpt runs/ldm_7tcdm3d/stage2_best.pt
    # validation split only
    python scripts/eval_ldm_7tcdm3d.py --val_only
    # metrics only, no per-sample PNGs (fast)
    python scripts/eval_ldm_7tcdm3d.py --no-grids
    # single subject + NIfTI export (the old "predict" use)
    python scripts/eval_ldm_7tcdm3d.py --subject 2024_040 --save_nii

Output layout:
    <out_dir>/grids/<id>.png      per-sample 3×3 grids   (--save_grids, default on)
    <out_dir>/<id>_pred.nii.gz    predicted volume       (--save_nii)
    <out_dir>/metrics.csv         per-sample MAE/MSE/PSNR/SSIM
    <out_dir>/summary.json        run config + aggregate stats
Default out_dir: outputs/predict/<subject> for --subject, else outputs/eval/ldm_7tcdm3d.

Architecture / diffusion hyper-parameters are read from the checkpoint's embedded
'args', so you rarely need to override them. The frozen AE is resolved from the
run's stage1_best.pt (or an explicit --ae_ckpt).
"""

import argparse
import csv
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from moyamoya.dataset import PrePostFMRI
from moyamoya.data import reconstruct_val_split
from moyamoya.metrics import compute_metrics
from moyamoya.transform import ToChannelsFirstAndNormalize
from moyamoya.models.ldm_7tcdm3d import load_7tcdm3d_checkpoint


# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------

from moyamoya.viz import percentile_norm as _to_np


def _get_slices(vol_np: np.ndarray) -> dict:
    D, H, W = vol_np.shape
    return {
        "Axial":    vol_np[D // 2, :, :],
        "Coronal":  vol_np[:, H // 2, :],
        "Sagittal": vol_np[:, :, W // 2],
    }


def save_grid(
    x_np: np.ndarray,
    y_np: np.ndarray,
    pred_np: np.ndarray,
    sample_id: str,
    metrics: dict,
    save_path: Path,
) -> None:
    xs = _get_slices(x_np)
    ys = _get_slices(y_np)
    ps = _get_slices(pred_np)

    views = ["Axial", "Coronal", "Sagittal"]
    cols  = ["Pre-surgery", "Prediction", "Post-surgery GT"]

    fig, axes = plt.subplots(3, 3, figsize=(10, 10), constrained_layout=True)

    for r, view in enumerate(views):
        for c, (img, title) in enumerate(zip(
            [xs[view], ps[view], ys[view]], cols
        )):
            ax = axes[r, c]
            ax.imshow(img, cmap="gray", origin="lower", vmin=0, vmax=1)
            ax.axis("off")
            if r == 0:
                ax.set_title(title, fontsize=11)
            if c == 0:
                ax.set_ylabel(view, fontsize=10)

    fig.suptitle(
        f"ID: {sample_id}  |  MAE={metrics['mae']:.4f}  MSE={metrics['mse']:.4f}"
        f"  PSNR={metrics['psnr']:.2f}  SSIM={metrics['ssim']:.4f}",
        fontsize=11,
    )
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_nifti(pred_b: torch.Tensor, pre_path: str, out_path: Path) -> tuple:
    """Write the predicted volume back to NIfTI in the source geometry.

    The transform did (X,Y,Z) -> permute(2,1,0) -> (Z,Y,X); invert it to (X,Y,Z)
    and reuse the source pre-surgery affine/header so the volume overlays the
    original scan. Intensities are in z-score-normalised units.
    """
    import nibabel as nib

    pred_zyx = pred_b.squeeze().cpu().float().numpy()                 # (Z,Y,X)
    pred_xyz = np.ascontiguousarray(np.transpose(pred_zyx, (2, 1, 0))).astype(np.float32)
    pre_img  = nib.load(pre_path)
    if pred_xyz.shape != pre_img.shape:
        raise SystemExit(f"shape mismatch {pred_xyz.shape} vs source {pre_img.shape}")
    hdr = pre_img.header.copy()
    hdr.set_data_dtype(np.float32)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(pred_xyz, affine=pre_img.affine, header=hdr), out_path)
    return pred_xyz.shape


def _subject_seed(base: int, sid: str) -> int:
    """Stable per-subject RNG seed so a subject's prediction is reproducible
    and independent of dataset ordering or batch-vs-single mode."""
    h = int(hashlib.sha1(sid.encode()).hexdigest()[:8], 16)
    return (base + h) % (2**31 - 1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root",      type=str,   default="fmri")
    p.add_argument("--ckpt",           type=str,   default="runs/ldm_7tcdm3d/stage2_best.pt")
    p.add_argument("--ae_ckpt",        type=str,   default=None,
                   help="Override the frozen AE checkpoint (default: resolved from the run's stage1_best.pt)")
    p.add_argument("--out_dir",        type=str,   default=None,
                   help="Default: outputs/predict/<subject> for --subject, else outputs/eval/ldm_7tcdm3d")
    p.add_argument("--pre_dirname",    type=str,   default="pre_surgery")
    p.add_argument("--post_dirname",   type=str,   default="6_months_post_surgery")
    # what to evaluate
    p.add_argument("--subject",        type=str,   default=None,
                   help="Evaluate a single subject id (e.g. 2024_040) instead of the whole set")
    p.add_argument("--val_only",       action="store_true", default=False,
                   help="Evaluate only the validation split (uses val_frac and split seed from the checkpoint)")
    p.add_argument("--val_frac",       type=float, default=None,
                   help="Val fraction (default: from checkpoint args, fallback 0.15)")
    p.add_argument("--seed",           type=int,   default=None,
                   help="Split seed for --val_only (default: from checkpoint args, fallback 42)")
    # sampling
    p.add_argument("--ddim_steps",     type=int,   default=None)
    p.add_argument("--guidance_scale", type=float, default=None)
    p.add_argument("--eta",            type=float, default=0.0)
    p.add_argument("--n_samples",      type=int,   default=1,
                   help="Average this many independent DDIM draws per subject (ensemble mean)")
    p.add_argument("--sample_seed",    type=int,   default=0,
                   help="Base seed for DDIM sampling noise (makes predictions reproducible)")
    p.add_argument("--amp",            action="store_true", default=True)
    # outputs
    p.add_argument("--save_grids",  dest="save_grids", action="store_true",
                   help="Render a per-sample 3×3 grid PNG (default on)")
    p.add_argument("--no-grids",    dest="save_grids", action="store_false",
                   help="Skip per-sample grids (metrics only — much faster on the full set)")
    p.set_defaults(save_grids=True)
    p.add_argument("--save_nii",       action="store_true", default=False,
                   help="Export each prediction as <id>_pred.nii.gz in the source geometry")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp    = bool(args.amp and device.type == "cuda")

    if args.out_dir:
        out_dir = Path(args.out_dir)
    elif args.subject:
        out_dir = Path(f"outputs/predict/{args.subject}")
    else:
        out_dir = Path("outputs/eval/ldm_7tcdm3d")
    out_dir.mkdir(parents=True, exist_ok=True)
    grids_dir = out_dir / "grids"

    model, raw = load_7tcdm3d_checkpoint(args.ckpt, device=device, ae_ckpt=args.ae_ckpt)
    for p_ in model.parameters():
        p_.requires_grad_(False)
    ckpt_args = raw["args"]

    ddim_steps     = args.ddim_steps     or ckpt_args.get("ddim_steps",     50)
    guidance_scale = args.guidance_scale or ckpt_args.get("guidance_scale",  3.0)

    print(f"Loaded: {args.ckpt}")
    print(f"  ddim_steps={ddim_steps}  guidance_scale={guidance_scale}  "
          f"n_samples={args.n_samples}  sample_seed={args.sample_seed}")

    transform = ToChannelsFirstAndNormalize(nonzero_mask=True)
    ds = PrePostFMRI(
        root_dir    = args.data_root,
        pre_dirname = args.pre_dirname,
        post_dirname= args.post_dirname,
        strict      = False,
        transform   = transform,
        return_paths= True,
    )

    # ── choose which samples to evaluate ──────────────────────────────────────
    if args.subject:
        try:
            sidx = next(i for i, (n, _, _) in enumerate(ds.pairs)
                        if n.replace(".nii.gz", "") == args.subject)
        except StopIteration:
            raise SystemExit(f"Subject {args.subject} not found in {args.data_root}")
        dataset, indices = ds, [sidx]
        print(f"Single-subject mode: {args.subject}")
    elif args.val_only:
        val_frac = args.val_frac or ckpt_args.get("val_frac", 0.15)
        seed     = args.seed     or ckpt_args.get("seed",     42)
        # Reconstruct the same fold this checkpoint trained against. Old (pre
        # k-fold) checkpoints have no "fold" key, so .get() → None → holdout.
        n_folds  = ckpt_args.get("n_folds")
        fold     = ckpt_args.get("fold")
        _, dataset = reconstruct_val_split(ds, val_frac, seed, n_folds=n_folds, fold=fold)
        indices = range(len(dataset))
        if fold is not None:
            print(f"Val-only mode: {len(dataset)} val samples (fold {fold}/{n_folds}, seed={seed})")
        else:
            print(f"Val-only mode: {len(dataset)} val samples (val_frac={val_frac}, seed={seed})")
    else:
        dataset, indices = ds, range(len(ds))

    indices = list(indices)
    print(f"Evaluating {len(indices)} sample(s)  |  Output: {out_dir}\n")

    rows = []
    for n_done, idx in enumerate(indices, start=1):
        x, y, meta = dataset[idx]
        sid = meta["id"]

        x_b = x.unsqueeze(0).to(device)
        y_b = y.unsqueeze(0).to(device)

        # seed once per subject, then draw the ensemble sequentially so the
        # draws differ but the whole set is reproducible for this subject
        seed = _subject_seed(args.sample_seed, sid)
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)

        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            draws = [model.generate(x_b, ddim_steps=ddim_steps, eta=args.eta,
                                    guidance_scale=guidance_scale)
                     for _ in range(args.n_samples)]
        pred_b   = draws[0] if len(draws) == 1 else torch.stack(draws, dim=0).mean(dim=0)
        pred_cpu = pred_b.squeeze(0).cpu()

        mask = (x != 0) | (y != 0)
        m = compute_metrics(pred_cpu, y, mask)

        if args.save_grids:
            save_grid(_to_np(x), _to_np(y), _to_np(pred_cpu), sid, m, grids_dir / f"{sid}.png")
        if args.save_nii:
            shape = save_nifti(pred_b, meta["pre_path"], out_dir / f"{sid}_pred.nii.gz")
            print(f"    → saved {sid}_pred.nii.gz  shape={shape}")

        rows.append({"id": sid, **m})
        print(f"[{n_done:>3}/{len(indices)}] {sid}  "
              f"MAE={m['mae']:.4f}  MSE={m['mse']:.4f}  "
              f"PSNR={m['psnr']:.2f}  SSIM={m['ssim']:.4f}")

    csv_path = out_dir / "metrics.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "mae", "mse", "psnr", "ssim"])
        writer.writeheader()
        writer.writerows(rows)

    def _stats(vals):
        a = np.asarray(vals, dtype=float)
        return {
            "mean":   float(a.mean()),
            "std":    float(a.std()),
            "median": float(np.median(a)),
            "min":    float(a.min()),
            "max":    float(a.max()),
        }

    print(f"\n{'='*60}")
    print(f"Samples: {len(rows)}")
    for col, label in [("mae", "MAE"), ("mse", "MSE"), ("psnr", "PSNR (dB)"), ("ssim", "SSIM")]:
        vals = [r[col] for r in rows]
        print(f"  {label:10s}  mean={np.mean(vals):.5f}  std={np.std(vals):.5f}  median={np.median(vals):.5f}")
    print(f"Metrics saved: {csv_path}")

    # aggregate summary — the durable, structured record of this eval run
    summary = {
        "model":          "ldm_7tcdm3d",
        "timestamp":      datetime.now().isoformat(),
        "ckpt":           args.ckpt,
        "data_root":      args.data_root,
        "subject":        args.subject,
        "val_only":       args.val_only,
        "ddim_steps":     ddim_steps,
        "guidance_scale": guidance_scale,
        "eta":            args.eta,
        "ensemble":       args.n_samples,
        "sample_seed":    args.sample_seed,
        "save_grids":     args.save_grids,
        "save_nii":       args.save_nii,
        "n_evaluated":    len(rows),
        "metrics":        {col: _stats([r[col] for r in rows])
                           for col in ["mae", "mse", "psnr", "ssim"]},
    }
    summary_path = out_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved: {summary_path}")


if __name__ == "__main__":
    main()
