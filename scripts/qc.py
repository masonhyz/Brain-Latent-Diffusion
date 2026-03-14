"""
save_pair_plots.py

Save a comparison plot for EVERY (pre, post) pair in PrePostFMRI:
- loads dataset with your training-time transform
- for each sample, saves a 3x2 montage of orthogonal mid-slices:
    rows: axial / coronal / sagittal
    cols: pre / post GT

Usage:
  python save_pair_plots.py \
    --data_root /path/to/fmri \
    --out_dir ./pair_plots \
    --max_items -1 \
    --pre_dirname pre_surgery \
    --post_dirname 6_months_post_surgery \
    --strict \
    --nonzero_mask \
    --print_every 50
"""

import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from typing import Dict

import numpy as np
import torch
import matplotlib.pyplot as plt

from moyamoya.dataset import PrePostFMRI
from moyamoya.transform import ToChannelsFirstAndNormalize


def get_orthogonal_slices(vol_chdhw: np.ndarray) -> Dict[str, np.ndarray]:
    """
    vol_chdhw: [C, D, H, W]
    Returns axial/coronal/sagittal mid-slices using channel 0.
    """
    v = vol_chdhw[0]  # [D,H,W]
    D, H, W = v.shape
    axial = v[D // 2, :, :]                 # (H,W)
    coronal = np.flipud(v[:, H // 2, :])    # (D,W)
    sagittal = np.flipud(v[:, :, W // 2])   # (D,H)
    return {"axial": axial, "coronal": coronal, "sagittal": sagittal}


def imshow_robust(ax, img: np.ndarray, title: str):
    finite = np.isfinite(img)
    if finite.sum() == 0:
        ax.imshow(np.zeros_like(img), cmap="gray")
        ax.set_title(title + " (no finite)", fontsize=9)
        ax.axis("off")
        return
    vmin, vmax = np.percentile(img[finite], [1, 99])
    ax.imshow(img, cmap="gray", vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=9)
    ax.axis("off")


def save_pair_figure(x_chdhw: np.ndarray, y_chdhw: np.ndarray, meta: Dict, out_path: Path):
    xs = get_orthogonal_slices(x_chdhw)
    ys = get_orthogonal_slices(y_chdhw)

    rows = ["axial", "coronal", "sagittal"]

    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(7, 9), constrained_layout=True)

    for r, view in enumerate(rows):
        imshow_robust(axes[r, 0], xs[view], f"{view.capitalize()} | Pre")
        imshow_robust(axes[r, 1], ys[view], f"{view.capitalize()} | Post (GT)")

    sid = meta.get("id", "")
    fn = meta.get("filename", "")
    fig.suptitle(f"ID: {sid} | {fn}", fontsize=11)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, required=True)
    p.add_argument("--pre_dirname", type=str, default="pre_surgery")
    p.add_argument("--post_dirname", type=str, default="6_months_post_surgery")
    p.add_argument("--strict", action="store_true", default=True)

    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--max_items", type=int, default=-1, help="-1 means all items")
    p.add_argument("--print_every", type=int, default=50)

    # keep this aligned with training
    p.add_argument("--nonzero_mask", action="store_true", default=True,
                   help="Match ToChannelsFirstAndNormalize(nonzero_mask=...)")
    args = p.parse_args()

    transform = ToChannelsFirstAndNormalize(nonzero_mask=bool(args.nonzero_mask))

    ds = PrePostFMRI(
        root_dir=args.data_root,
        pre_dirname=args.pre_dirname,
        post_dirname=args.post_dirname,
        strict=args.strict,
        transform=transform,
        return_paths=True,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    n = len(ds) if args.max_items < 0 else min(len(ds), args.max_items)

    for i in range(n):
        x, y, meta = ds[i]  # torch [C,D,H,W]
        x_np = x.detach().cpu().numpy()
        y_np = y.detach().cpu().numpy()

        # Make filenames stable + informative
        sid = str(meta.get("id", ""))
        fn = str(meta.get("filename", ""))
        fn_safe = fn.replace("/", "_").replace("\\", "_").replace(" ", "_")

        out_path = out_dir / f"{i:05d}_id-{sid}_file-{fn_safe}.png"
        save_pair_figure(x_np, y_np, meta, out_path)

        if (i + 1) % args.print_every == 0 or i == 0 or (i + 1) == n:
            print(f"[{i+1:>6}/{n}] saved {out_path.name}")

    print(f"\nDone. Wrote {n} plots to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
