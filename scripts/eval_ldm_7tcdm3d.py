"""
eval_ldm_7tcdm3d.py

Evaluate the 7TCDM-3D Latent Diffusion stage-2 checkpoint on all samples
in the PrePostFMRI dataset.  For each sample produces a 3×3 grid:

    rows : Axial | Coronal | Sagittal
    cols : Pre-surgery | Post-surgery GT | Prediction

Also writes outputs/eval_ldm_7tcdm3d/metrics.csv with per-sample metrics
(MAE, MSE, PSNR, SSIM) and prints a summary.

Usage:
    python scripts/eval_ldm_7tcdm3d.py \\
        --data_root fmri \\
        --ckpt runs/ldm_7tcdm3d/stage2_best.pt \\
        --out_dir outputs/eval_ldm_7tcdm3d

All architecture / diffusion hyper-parameters are read from the checkpoint's
embedded 'args' dict, so you rarely need to override them.
"""

import argparse
import csv
import sys
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
from moyamoya.models.ldm_7tcdm3d import build_paired_latent_diffusion_7tcdm


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


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root",      type=str,   default="fmri")
    p.add_argument("--ckpt",           type=str,   default="runs/ldm_7tcdm3d/stage2_best.pt")
    p.add_argument("--out_dir",        type=str,   default="outputs/eval_ldm_7tcdm3d")
    p.add_argument("--pre_dirname",    type=str,   default="pre_surgery")
    p.add_argument("--post_dirname",   type=str,   default="6_months_post_surgery")
    p.add_argument("--ddim_steps",     type=int,   default=None)
    p.add_argument("--guidance_scale", type=float, default=None)
    p.add_argument("--eta",            type=float, default=0.0)
    p.add_argument("--amp",            action="store_true", default=True)
    p.add_argument("--val_only",       action="store_true", default=False,
                   help="Evaluate only on the validation split (uses val_frac and seed from checkpoint)")
    p.add_argument("--val_frac",       type=float, default=None,
                   help="Val fraction (default: from checkpoint args, fallback 0.15)")
    p.add_argument("--seed",           type=int,   default=None,
                   help="Split seed (default: from checkpoint args, fallback 42)")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp    = bool(args.amp and device.type == "cuda")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    ckpt_args = raw["args"]

    ddim_steps     = args.ddim_steps     or ckpt_args.get("ddim_steps",     50)
    guidance_scale = args.guidance_scale or ckpt_args.get("guidance_scale",  3.0)

    model = build_paired_latent_diffusion_7tcdm(
        z_channels       = ckpt_args["z_channels"],
        embed_dim        = ckpt_args["embed_dim"],
        ae_ch            = ckpt_args["ae_ch"],
        ae_res_blocks    = ckpt_args["ae_res_blocks"],
        ae_resolution    = ckpt_args["ae_resolution"],
        kl_weight        = ckpt_args["kl_weight"],
        diff_dim         = ckpt_args["diff_dim"],
        diff_dim_mults   = tuple(ckpt_args["diff_dim_mults"]),
        init_kernel_size = ckpt_args["init_kernel_size"],
        resnet_groups    = ckpt_args["resnet_groups"],
        T                = ckpt_args["T"],
    ).to(device)

    model.ae.load_state_dict(raw["ae"])
    model.denoiser.load_state_dict(raw["denoiser"])
    model.eval()
    for p_ in model.parameters():
        p_.requires_grad_(False)

    print(f"Loaded: {args.ckpt}")
    print(f"  ddim_steps={ddim_steps}  guidance_scale={guidance_scale}")

    transform = ToChannelsFirstAndNormalize(nonzero_mask=True)
    ds = PrePostFMRI(
        root_dir    = args.data_root,
        pre_dirname = args.pre_dirname,
        post_dirname= args.post_dirname,
        strict      = False,
        transform   = transform,
        return_paths= True,
    )

    if args.val_only:
        val_frac = args.val_frac or ckpt_args.get("val_frac", 0.15)
        seed     = args.seed     or ckpt_args.get("seed",     42)
        _, val_subset = reconstruct_val_split(ds, val_frac, seed)
        eval_set = val_subset
        print(f"Val-only mode: {len(val_subset)} val samples (val_frac={val_frac}, seed={seed})")
    else:
        eval_set = ds

    print(f"Evaluating {len(eval_set)} samples  |  Output: {out_dir}\n")

    rows = []
    for idx in range(len(eval_set)):
        x, y, meta = eval_set[idx]
        sid = meta["id"]

        x_b = x.unsqueeze(0).to(device)
        y_b = y.unsqueeze(0).to(device)

        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            pred_b = model.generate(
                x_b,
                ddim_steps    = ddim_steps,
                eta           = args.eta,
                guidance_scale= guidance_scale,
            )

        pred_cpu = pred_b.squeeze(0).cpu()
        x_cpu    = x
        y_cpu    = y
        mask     = (x_cpu != 0) | (y_cpu != 0)

        m = compute_metrics(pred_cpu, y_cpu, mask)

        grid_path = out_dir / f"{sid}.png"
        save_grid(_to_np(x_cpu), _to_np(y_cpu), _to_np(pred_cpu), sid, m, grid_path)

        rows.append({"id": sid, **m})
        print(f"[{idx+1:>3}/{len(eval_set)}] {sid}  "
              f"MAE={m['mae']:.4f}  MSE={m['mse']:.4f}  "
              f"PSNR={m['psnr']:.2f}  SSIM={m['ssim']:.4f}")

    csv_path = out_dir / "metrics.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "mae", "mse", "psnr", "ssim"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n{'='*60}")
    print(f"Samples: {len(rows)}")
    for col, label in [("mae", "MAE"), ("mse", "MSE"), ("psnr", "PSNR (dB)"), ("ssim", "SSIM")]:
        vals = [r[col] for r in rows]
        print(f"  {label:10s}  mean={np.mean(vals):.5f}  std={np.std(vals):.5f}  median={np.median(vals):.5f}")
    print(f"Metrics saved: {csv_path}")


if __name__ == "__main__":
    main()
