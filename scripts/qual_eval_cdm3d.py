"""
Qualitative evaluation of a trained CDM3D checkpoint on the validation split.

For every subject in the held-out validation set (reconstructed with the exact
train/val split used at training time), runs DDIM sampling with the EMA denoiser
and saves a 3×3 grid:

    rows  = Axial / Coronal / Sagittal mid-slices
    cols  = Pre-surgery / Prediction / Post-surgery GT

One PNG per validation subject, named <subject>.png, plus a printed MAE/SSIM so
the grids can be skimmed worst-to-best.

Usage (from project root):
    python scripts/qual_eval_cdm3d.py                       # runs/cdm3d/best.pt
    python scripts/qual_eval_cdm3d.py --run cdm3d_base      # runs/cdm3d_base/best.pt
    python scripts/qual_eval_cdm3d.py --run cdm3d_base --ckpt_name last.pt
    python scripts/qual_eval_cdm3d.py --ckpt runs/cdm3d/best.pt --ddim_steps 50
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from skimage.metrics import structural_similarity
from torch.utils.data import random_split

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from moyamoya.dataset import PrePostFMRI
from moyamoya.transform import ToChannelsFirstAndNormalize
from moyamoya.utils import seed_everything, get_device
from moyamoya.models.cdm3d import build_cdm3d

# Reuse the training visualisation so qualitative grids match exactly.
from scripts.train_cdm3d import _to_np, save_grid, compute_metrics


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root",      type=str, default="fmri")
    p.add_argument("--run",            type=str, default="cdm3d",
                   help="Run name under runs/ (e.g. cdm3d_base, cdm3d_small)")
    p.add_argument("--ckpt_name",      type=str, default="best.pt",
                   help="Checkpoint file within the run dir (e.g. best.pt, last.pt)")
    p.add_argument("--ckpt",           type=str, default=None,
                   help="Full checkpoint path; overrides --run/--ckpt_name when set")
    p.add_argument("--out_dir",        type=str, default=None,
                   help="Output dir (default: runs/<run>/qual)")
    p.add_argument("--ddim_steps",     type=int, default=None,
                   help="Override DDIM steps (default: value stored in ckpt args)")
    p.add_argument("--guidance_scale", type=float, default=None,
                   help="Override CFG scale (default: value stored in ckpt args)")
    args = p.parse_args()
    if args.ckpt is None:
        args.ckpt = str(Path("runs") / args.run / args.ckpt_name)
    if args.out_dir is None:
        args.out_dir = str(Path(args.ckpt).resolve().parent / "qual")
    return args


def main():
    args = get_args()
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    a = ckpt["args"]

    seed       = a.get("seed", 42)
    val_frac   = a.get("val_frac", 0.15)
    ddim_steps = args.ddim_steps     if args.ddim_steps     is not None else a["ddim_steps"]
    gs         = args.guidance_scale if args.guidance_scale is not None else a["guidance_scale"]

    seed_everything(seed)
    device = get_device()
    print(f"Device: {device}")
    print(f"Checkpoint: {args.ckpt}  (epoch {ckpt.get('epoch', '?')})")
    print(f"DDIM steps: {ddim_steps}   guidance_scale: {gs}\n")

    # ── data: reconstruct the exact split used at training time ────────────────
    base_tfm = ToChannelsFirstAndNormalize(nonzero_mask=True)
    ds = PrePostFMRI(root_dir=args.data_root, transform=base_tfm, strict=False)

    n_val   = max(1, int(len(ds) * val_frac))
    n_train = len(ds) - n_val
    g = torch.Generator().manual_seed(seed)
    _, val_subset = random_split(ds, [n_train, n_val], generator=g)
    val_indices = list(val_subset.indices)
    print(f"Dataset: {len(ds)} subjects  (train={n_train}, val={n_val})\n")

    # ── model: EMA weights (same as training-time validation) ──────────────────
    model = build_cdm3d(
        dim=a["dim"],
        dim_mults=tuple(a["dim_mults"]),
        init_kernel_size=a["init_kernel_size"],
        resnet_groups=a["resnet_groups"],
        timesteps=a["T"],
        ddim_steps=ddim_steps,
        ssim_weight=a["ssim_weight"],
        cfg_drop_prob=a["cfg_drop_prob"],
    ).to(device)
    model.load_state_dict(ckpt["ema_model"])
    model.eval()
    print(f"[CDM3D] EMA params loaded: {sum(p.numel() for p in model.parameters()):,}\n")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── per-subject qualitative grids ──────────────────────────────────────────
    print(f"Saving 3×3 grids for {len(val_indices)} validation subjects → {out_dir}\n")
    records = []
    for i, idx in enumerate(val_indices):
        x, y = ds[idx]
        subject = ds.pairs[idx][0].replace(".nii.gz", "")
        x_b = x.unsqueeze(0).to(device)
        y_b = y.unsqueeze(0).to(device)
        with torch.no_grad():
            pred = model.sample(x_b, guidance_scale=gs)

        mask = (x_b != 0) | (y_b != 0)
        m = compute_metrics(pred.float().cpu(), y_b, mask)

        title = f"{subject}   MAE={m['mae']:.3f}  SSIM={m['ssim']:.3f}"
        save_grid(
            _to_np(x_b), _to_np(y_b), _to_np(pred),
            title, out_dir / f"{subject}.png",
        )
        records.append((subject, m["mae"], m["ssim"]))
        print(f"  [{i + 1:>3}/{len(val_indices)}] {subject}  "
              f"MAE={m['mae']:.4f}  SSIM={m['ssim']:.4f}")

    # ── summary, sorted worst→best by MAE ──────────────────────────────────────
    records.sort(key=lambda r: r[1], reverse=True)
    print("\n" + "=" * 52)
    print("Validation subjects, worst→best by MAE")
    print("=" * 52)
    for subject, mae, ssim in records:
        print(f"  {subject:<20} MAE={mae:.4f}  SSIM={ssim:.4f}")
    print("=" * 52)
    print(f"\nSaved {len(records)} grids to {out_dir}")


if __name__ == "__main__":
    main()
