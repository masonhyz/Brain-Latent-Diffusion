#!/usr/bin/env python3
"""
infer_cyclegan_2d.py
────────────────────
Batch inference and visual comparison for the trained 2D fMRI CycleGAN.

Model architecture is read automatically from
<checkpoints_dir>/<name>/train_opt.txt so the generator is always built
with the exact same spec as was used during training.

For every pre-surgery volume, picks 3 evenly-spaced axial slices from the
middle 60% of the Z axis (matching training data) and passes each through
the trained G_A (pre→post) generator.  Saves a comparison figure:

    rows : 3 axial slices (evenly spaced in middle 60% of Z)
    cols : real_A (pre)  |  fake_B (generated post)  |  real_B (actual post)

The paired real_B is shown for reference only — training was unpaired so
there is no guarantee of subject-level correspondence.

Usage
─────
    # latest checkpoint, default paths
    python infer_cyclegan_2d.py

    # specific epoch, custom output directory
    python infer_cyclegan_2d.py --epoch 100 --output_dir outputs/epoch100

    # only run a few subjects to check quickly
    python infer_cyclegan_2d.py --max_subjects 10

    # show more or fewer sample slices per subject
    python infer_cyclegan_2d.py --n_slices 5
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
REPO = HERE.parent / "third_party" / "pytorch-CycleGAN-and-pix2pix"
FMRI_ROOT = HERE.parent / "fmri"
CHECKPOINTS = HERE.parent / "checkpoints"

sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(REPO))
from models.networks import define_G  # noqa: E402 – must come after sys.path edit


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Batch CycleGAN inference: plot real_A / fake_B / real_B per subject",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--name",            default="fmri_cyclegan_2d",
                   help="Experiment name (checkpoint subdirectory)")
    p.add_argument("--epoch",           default="latest",
                   help="Checkpoint to load: 'latest' or an integer e.g. '100'")
    p.add_argument("--checkpoints_dir", default=str(CHECKPOINTS))
    p.add_argument("--dataroot",        default=str(FMRI_ROOT))
    p.add_argument("--pre_dirname",     default="pre_surgery")
    p.add_argument("--post_dirname",    default="6_months_post_surgery")
    p.add_argument("--output_dir",      default=None,
                   help="Where to save PNGs (default: <checkpoints_dir>/<name>/comparisons/)")
    p.add_argument("--slice_size",      type=int, default=None,
                   help="Override the slice_size from train_opt.txt")
    p.add_argument("--n_slices",        type=int, default=3,
                   help="Number of evenly-spaced axial slices to show per subject "
                        "(drawn from the middle 60%% of Z, same as training)")
    p.add_argument("--max_subjects",    type=int, default=None,
                   help="Stop after this many subjects (useful for quick checks)")
    p.add_argument("--device",          default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


# ── train_opt.txt parser ───────────────────────────────────────────────────────

def load_train_opt(opt_path: Path) -> dict:
    """
    Parse <checkpoints_dir>/<name>/train_opt.txt into a plain dict.
    Lines look like:
                       netG: resnet_9blocks                	[default: ...]
    Returns {'netG': 'resnet_9blocks', ...} with string values.
    """
    opts = {}
    with open(opt_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("---"):
                continue
            # strip optional trailing "[default: ...]"
            line = line.split("\t[default:")[0].rstrip()
            if ":" not in line:
                continue
            key, _, val = line.partition(":")
            opts[key.strip()] = val.strip()
    return opts


# ── Model ─────────────────────────────────────────────────────────────────────

def load_generator(
    checkpoint_path: Path, train_opts: dict, device: str
) -> torch.nn.Module:
    netG      = train_opts.get("netG",      "resnet_9blocks")
    ngf       = int(train_opts.get("ngf",       64))
    norm      = train_opts.get("norm",      "instance")
    input_nc  = int(train_opts.get("input_nc",  1))
    output_nc = int(train_opts.get("output_nc", 1))
    init_type = train_opts.get("init_type", "normal")
    init_gain = float(train_opts.get("init_gain", 0.02))
    no_dropout = train_opts.get("no_dropout", "False").strip() == "True"

    net = define_G(
        input_nc=input_nc, output_nc=output_nc, ngf=ngf, netG=netG,
        norm=norm, use_dropout=(not no_dropout),
        init_type=init_type, init_gain=init_gain,
    )
    state = torch.load(str(checkpoint_path), map_location="cpu")
    net.load_state_dict(state)
    return net.eval().to(device)


# ── Volume helpers (mirrors FmriSliceDataset normalisation) ───────────────────

def load_and_normalize(path: Path) -> torch.Tensor:
    """NIfTI → (X, Y, Z) float32, z-scored on brain voxels, scaled to [-1, 1].

    Brain mask: voxels > 1% of volume maximum.  Mirrors FmriSliceDataset._normalize.
    Using a threshold (not vol != 0) avoids including float32 background residuals
    in the statistics, which would corrupt mean/std for both domains differently.
    """
    vol = torch.from_numpy(nib.load(str(path)).get_fdata(dtype="float32"))
    threshold = vol.max() * 0.01
    mask = vol > threshold
    if mask.any():
        vals = vol[mask]
        mean = vals.mean()
        std = vals.std(unbiased=False).clamp_min(1e-6)
        vol = (vol - mean) / std
    return vol.clamp(-3.0, 3.0) / 3.0


def middle_z_indices(n_z: int, n_slices: int) -> List[int]:
    """
    Return n_slices evenly-spaced Z indices from the middle 60% of [0, n_z).
    Window: [0.20·n_z, 0.80·n_z) — mirrors FmriSliceDataset._middle_z_indices.
    """
    lo = int(round(n_z * 0.20))
    hi = int(round(n_z * 0.80))
    hi = max(hi, lo + 1)
    return [int(x) for x in np.linspace(lo, hi - 1, n_slices, dtype=float).round()]


def axial_slice(vol: torch.Tensor, z_idx: int, size: int) -> torch.Tensor:
    """
    Extract axial slice at z_idx, resize to size×size.
    Returns shape (1, 1, size, size) – ready for the network or display.
    """
    sl = vol[:, :, z_idx].unsqueeze(0).unsqueeze(0)    # (1, 1, X, Y)
    if sl.shape[-2:] != (size, size):
        sl = F.interpolate(sl, size=(size, size), mode="bilinear", align_corners=False)
    return sl                                           # (1, 1, size, size)


def to_display(t: torch.Tensor) -> np.ndarray:
    """(1, 1, H, W) tensor in [-1, 1] → (H, W) float array in [0, 1]."""
    return ((t.squeeze().cpu().float().numpy() + 1.0) / 2.0).clip(0.0, 1.0)


# ── Plotting ──────────────────────────────────────────────────────────────────

def make_figure(
    subject_id: str,
    z_indices: List[int],
    real_A_vol: torch.Tensor,
    fake_B_slices: List[torch.Tensor],      # one (1,1,H,W) per z index, G_A output
    real_B_vol: Optional[torch.Tensor],
    slice_size: int,
) -> plt.Figure:
    """
    n_slices rows × 3 cols grid (or n_slices×2 if no paired real_B exists).

    Rows : evenly-spaced axial slices from middle 60% of Z
    Cols : real_A  |  fake_B  |  real_B (if available)
    """
    n_rows = len(z_indices)
    has_B = real_B_vol is not None
    n_cols = 3 if has_B else 2
    col_titles = ["real_A  (pre-surgery)", "fake_B  (generated post)", "real_B  (actual post)"]

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.5 * n_cols, 4.5 * n_rows))
    if n_rows == 1:
        axes = axes[np.newaxis, :]          # ensure 2-D indexing always works
    fig.suptitle(f"Subject: {subject_id}", fontsize=13, fontweight="bold", y=1.01)

    for row, (z_idx, fake_sl) in enumerate(zip(z_indices, fake_B_slices)):
        panels = [
            to_display(axial_slice(real_A_vol, z_idx, slice_size)),
            to_display(fake_sl),
        ]
        if has_B:
            panels.append(to_display(axial_slice(real_B_vol, z_idx, slice_size)))

        for col, img in enumerate(panels):
            ax = axes[row, col]
            ax.imshow(img, cmap="gray", vmin=0.0, vmax=1.0)
            ax.axis("off")
            if row == 0:
                ax.set_title(col_titles[col], fontsize=10, pad=6)
            if col == 0:
                ax.set_ylabel(f"z={z_idx}", fontsize=9, labelpad=4)

    fig.tight_layout()
    return fig


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    checkpoint_path = Path(args.checkpoints_dir) / args.name / f"{args.epoch}_net_G_A.pth"
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}\n"
            f"Available epochs: "
            + ", ".join(
                p.stem.replace("_net_G_A", "")
                for p in sorted((Path(args.checkpoints_dir) / args.name).glob("*_net_G_A.pth"))
            )
        )

    output_dir = (
        Path(args.output_dir) if args.output_dir
        else Path(args.checkpoints_dir) / args.name / "comparisons"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    opt_path = Path(args.checkpoints_dir) / args.name / "train_opt.txt"
    if not opt_path.exists():
        raise FileNotFoundError(f"train_opt.txt not found: {opt_path}")
    train_opts = load_train_opt(opt_path)

    # slice_size: CLI override > train_opt.txt > fallback 256
    slice_size = args.slice_size or int(train_opts.get("slice_size", 256))

    print(f"Checkpoint : {checkpoint_path}")
    print(f"train_opt  : {opt_path}")
    print(f"Output dir : {output_dir}")
    print(f"Device     : {args.device}")
    print(f"netG={train_opts.get('netG','?')}  ngf={train_opts.get('ngf','?')}  "
          f"norm={train_opts.get('norm','?')}  no_dropout={train_opts.get('no_dropout','?')}  "
          f"slice_size={slice_size}\n")

    G_A = load_generator(checkpoint_path, train_opts, args.device)

    pre_dir = Path(args.dataroot) / args.pre_dirname
    post_dir = Path(args.dataroot) / args.post_dirname
    pre_files = sorted(pre_dir.glob("*.nii.gz"))
    post_map = {p.name: p for p in post_dir.glob("*.nii.gz")}

    if args.max_subjects is not None:
        pre_files = pre_files[: args.max_subjects]

    print(f"Processing {len(pre_files)} subjects ...\n")

    for nii_path in pre_files:
        subject_id = nii_path.name.replace(".nii.gz", "")
        print(f"  {subject_id}", end=" ... ", flush=True)

        real_A_vol = load_and_normalize(nii_path)

        # Pick evenly-spaced axial slices from the middle 60% of Z
        z_indices = middle_z_indices(real_A_vol.shape[2], args.n_slices)

        # Run each selected axial slice through G_A
        fake_B_slices: List[torch.Tensor] = []
        with torch.no_grad():
            for z in z_indices:
                sl = axial_slice(real_A_vol, z, slice_size).to(args.device)
                fake_B_slices.append(G_A(sl).cpu())

        # Paired real_B for the same subject (reference only – training was unpaired)
        real_B_vol = load_and_normalize(post_map[nii_path.name]) if nii_path.name in post_map else None

        fig = make_figure(subject_id, z_indices, real_A_vol, fake_B_slices, real_B_vol, slice_size)
        out_path = output_dir / f"{subject_id}.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"saved {out_path.name}")

    print(f"\nDone – {len(pre_files)} figures in {output_dir}/")


if __name__ == "__main__":
    main()
