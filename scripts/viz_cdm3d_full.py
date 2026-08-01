"""
Visualize trained CDM3D checkpoints over the WHOLE dataset (every subject, no
train/val split), using the plain vanilla pipeline — the same normalization and
sampling that produced the during-training grids in runs/<run>/vis/.

The displayed prediction is raw/vanilla — shown exactly as the denoiser emits it
(a faint textured background outside the brain is expected; that is the raw vanilla
output, same as vis/).  Masking is applied for EVALUATION ONLY: the MAE/SSIM are
brain-masked (background zeroed for scoring, never in the image) — matching the
"only make a difference numerically" intent.

For each subject it saves a 3x3 grid:
    rows = Axial / Coronal / Sagittal mid-slices
    cols = Pre-surgery / Prediction / Post-surgery GT
plus a printed MAE/SSIM table sorted worst -> best.

Usage (from project root):
    python scripts/viz_cdm3d_full.py                                   # both runs, best.pt
    python scripts/viz_cdm3d_full.py --runs cdm3d_base                 # a single run
    python scripts/viz_cdm3d_full.py --runs cdm3d_base cdm3d_small --ddim_steps 50
    python scripts/viz_cdm3d_full.py --ckpt_name last.pt              # final epoch instead of best
    GPU=0 python scripts/viz_cdm3d_full.py                            # choose GPU
"""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from moyamoya.dataset import PrePostFMRI
from moyamoya.transform import ToChannelsFirstAndNormalize
from moyamoya.utils import seed_everything, get_device
from moyamoya.models.cdm3d import build_cdm3d

# Reuse the exact training-time visualisation + metrics so the grids match vis/.
from scripts.train_cdm3d import _to_np, save_grid, compute_metrics


def get_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data_root",      type=str, default="fmri")
    p.add_argument("--runs",           type=str, nargs="+",
                   default=["cdm3d_base", "cdm3d_small"],
                   help="Run name(s) under runs/ to visualize.")
    p.add_argument("--ckpt_name",      type=str, default="best.pt",
                   help="Checkpoint file within each run dir (e.g. best.pt, last.pt).")
    p.add_argument("--out_name",       type=str, default="viz_full",
                   help="Sub-dir under runs/<run>/ to write the grids into.")
    p.add_argument("--ddim_steps",     type=int, default=None,
                   help="Override DDIM steps (default: value stored in ckpt args).")
    p.add_argument("--guidance_scale", type=float, default=None,
                   help="Override CFG scale (default: value stored in ckpt args).")
    p.add_argument("--seed",           type=int, default=42,
                   help="Seed for the DDIM sampling noise (reproducibility).")
    return p.parse_args()


def load_model(ckpt_path: Path, ddim_steps_override, gs_override, device):
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    a = ckpt["args"]
    ddim_steps = ddim_steps_override if ddim_steps_override is not None else a["ddim_steps"]
    gs         = gs_override         if gs_override         is not None else a["guidance_scale"]

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
    model.load_state_dict(ckpt["ema_model"])   # EMA weights, same as training-time validation
    model.eval()
    return model, ckpt, ddim_steps, gs


def visualize_run(run: str, args, ds: PrePostFMRI, device) -> None:
    ckpt_path = Path("runs") / run / args.ckpt_name
    if not ckpt_path.is_file():
        print(f"[skip] checkpoint not found: {ckpt_path}")
        return

    model, ckpt, ddim_steps, gs = load_model(ckpt_path, args.ddim_steps, args.guidance_scale, device)
    out_dir = Path("runs") / run / args.out_name
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== {run}  (epoch {ckpt.get('epoch', '?')}, ddim_steps={ddim_steps}, "
          f"guidance_scale={gs}) ===")
    print(f"Saving {len(ds)} grids -> {out_dir}")

    seed_everything(args.seed)   # deterministic DDIM noise across runs

    records = []
    for i in range(len(ds)):
        x, y = ds[i]
        subject = ds.pairs[i][0].replace(".nii.gz", "")
        x_b = x.unsqueeze(0).to(device)
        y_b = y.unsqueeze(0).to(device)

        with torch.no_grad():
            pred = model.sample(x_b, guidance_scale=gs)   # vanilla output, shown as-is

        # Brain-masked EVALUATION only (numerical, never visual).  Vanilla z-score
        # maps the exactly-zero raw background to the per-volume minimum, so
        # `!= min` recovers the brain.  All four metrics (incl. SSIM) are averaged
        # over this mask, so no background pre-zeroing is needed.
        brain = (x_b != x_b.min()) | (y_b != y_b.min())
        m = compute_metrics(pred.float().cpu(), y_b, brain)

        # Visualization stays raw/vanilla — same as runs/<run>/vis/.
        title = f"{subject}   MAE={m['mae']:.3f}  SSIM={m['ssim']:.3f}"
        save_grid(_to_np(x_b), _to_np(y_b), _to_np(pred), title, out_dir / f"{subject}.png")
        records.append((subject, m["mae"], m["ssim"]))
        print(f"  [{i + 1:>3}/{len(ds)}] {subject}  MAE={m['mae']:.4f}  SSIM={m['ssim']:.4f}")

    records.sort(key=lambda r: r[1], reverse=True)
    print("\n" + "=" * 52)
    print(f"{run}: subjects worst -> best by MAE")
    print("=" * 52)
    for subject, mae, ssim in records:
        print(f"  {subject:<20} MAE={mae:.4f}  SSIM={ssim:.4f}")
    print("=" * 52)
    print(f"Saved {len(records)} grids to {out_dir}")


def main():
    args = get_args()
    device = get_device()
    print(f"Device: {device}")

    # Whole dataset (no train/val split), vanilla normalization — same as vis/.
    tfm = ToChannelsFirstAndNormalize(nonzero_mask=True)
    ds = PrePostFMRI(root_dir=args.data_root, transform=tfm, strict=False)
    print(f"Dataset: {len(ds)} subjects (whole dataset, no split)")

    for run in args.runs:
        visualize_run(run, args, ds, device)


if __name__ == "__main__":
    main()
