"""
Evaluate a trained CDM3D checkpoint on the validation split and the full dataset.

Reconstructs the exact train/val split used at training time (same seed, val_frac,
random_split) so the "validation" numbers correspond to held-out subjects. Uses the
EMA denoiser weights for inference (same as training-time validation).

Metrics (brain-masked, z-score space, mask = pre!=0 | post!=0):
  MAE, MSE, PSNR, SSIM

Usage (from project root):
    python scripts/eval_cdm3d.py
    python scripts/eval_cdm3d.py --ckpt runs/cdm3d/best.pt --ddim_steps 50
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from moyamoya.dataset import PrePostFMRI
from moyamoya.transform import ToChannelsFirstAndNormalize
from moyamoya.data import reconstruct_val_split
from moyamoya.utils import seed_everything, get_device
from moyamoya.metrics import compute_metrics
from moyamoya.models.cdm3d import build_cdm3d




def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root",      type=str, default="fmri")
    p.add_argument("--ckpt",           type=str, default="runs/cdm3d/best.pt")
    p.add_argument("--out_dir",        type=str, default="outputs/eval_cdm3d")
    p.add_argument("--ddim_steps",     type=int, default=None,
                   help="Override DDIM steps (default: value stored in ckpt args)")
    p.add_argument("--guidance_scale", type=float, default=None,
                   help="Override CFG scale (default: value stored in ckpt args)")
    return p.parse_args()


def evaluate(model, indices, ds, guidance_scale, device, tag):
    """Run inference over a list of dataset indices; return per-subject records."""
    records = []
    n = len(indices)
    for i, idx in enumerate(indices):
        x, y = ds[idx]
        sample_id = ds.pairs[idx][0].replace(".nii.gz", "")
        x_b = x.unsqueeze(0).to(device)
        y_b = y.unsqueeze(0).to(device)
        with torch.no_grad():
            pred = model.sample(x_b, guidance_scale=guidance_scale)
        # Brain-masked eval: vanilla z-score puts the zero background at the volume
        # minimum, so `!= min` recovers the brain. Zero pred + target background so
        # the whole-volume SSIM stays brain-driven (MAE/MSE/PSNR use the mask).
        brain = (x_b != x_b.min()) | (y_b != y_b.min())
        m = compute_metrics((pred * brain).float().cpu(), y_b * brain, brain)
        m["subject"] = sample_id
        records.append(m)
        if (i + 1) % 10 == 0 or (i + 1) == n:
            run_mae = np.mean([r["mae"] for r in records])
            print(f"  [{tag}] {i + 1}/{n}  running MAE={run_mae:.4f}")
    return records


def summarize(records: list) -> dict:
    out = {}
    for k in ["mae", "mse", "psnr", "ssim"]:
        vals = np.array([r[k] for r in records])
        out[k] = {"mean": float(vals.mean()), "std": float(vals.std())}
    return out


def print_table(name: str, n: int, s: dict):
    print(f"\n{name}  (n={n})")
    print("-" * 52)
    for k, label in [("mae", "MAE"), ("mse", "MSE"),
                     ("psnr", "PSNR"), ("ssim", "SSIM")]:
        print(f"  {label:<5}: {s[k]['mean']:.4f} ± {s[k]['std']:.4f}")


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

    train_subset, val_subset = reconstruct_val_split(ds, val_frac, seed)
    val_indices  = list(val_subset.indices)
    full_indices = list(range(len(ds)))
    print(f"Dataset: {len(ds)} subjects  (train={len(train_subset)}, val={len(val_subset)})\n")

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
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[CDM3D] EMA params loaded: {n_params:,}\n")

    # ── validation split ───────────────────────────────────────────────────────
    print("Evaluating validation split…")
    val_records = evaluate(model, val_indices, ds, gs, device, "val")
    val_summary = summarize(val_records)

    # ── full dataset ───────────────────────────────────────────────────────────
    print("\nEvaluating full dataset…")
    full_records = evaluate(model, full_indices, ds, gs, device, "full")
    full_summary = summarize(full_records)

    # ── report ─────────────────────────────────────────────────────────────────
    print("\n" + "=" * 52)
    print("RESULTS — CDM3D (brain-masked, z-score space)")
    print("=" * 52)
    print_table("Validation split", len(val_records), val_summary)
    print_table("Full dataset",     len(full_records), full_summary)
    print("=" * 52)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "results.json", "w") as f:
        json.dump({
            "ckpt": args.ckpt,
            "epoch": ckpt.get("epoch"),
            "ddim_steps": ddim_steps,
            "guidance_scale": gs,
            "val": {"n": len(val_records), "summary": val_summary,
                    "per_subject": val_records},
            "full": {"n": len(full_records), "summary": full_summary,
                     "per_subject": full_records},
        }, f, indent=2)
    print(f"\nSaved: {out_dir / 'results.json'}")


if __name__ == "__main__":
    main()
