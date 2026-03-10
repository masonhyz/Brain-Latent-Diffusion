"""
infer_one.py

Inference on a single (pre, post) pair from your PrePostFMRI dataset:
- loads checkpoint
- builds UNet3D with flexible base_channels
- runs forward pass
- reports reconstruction loss (L1 by default; optional MSE)
- plots a chosen slice comparing: pre, post GT, pred

Usage:
  python infer_one.py \
    --data_root /path/to/fmri \
    --ckpt /path/to/best.pt \
    --base_channels 128 \
    --sample_idx 0 \
    --slice_axis D \
    --slice_index -1 \
    --loss l1 \
    --save_fig ./infer.png

Notes:
- Assumes you have dataset.py with PrePostFMRI, and your UNet3D + transform logic available.
- If your dataset volumes are 4D (X,Y,Z,T), the provided transform treats T as channels, matching your training.
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

from dataset import PrePostFMRI
from unet import UNet3D
from transform import ToChannelsFirstAndNormalize


def l1_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return (pred - target).abs().mean()


def mse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.mean((pred - target) ** 2)


def get_orthogonal_slices(vol_chdhw: np.ndarray):
    """
    vol_chdhw: [C, D, H, W]
    Returns correctly oriented axial, coronal, sagittal slices
    """
    v = vol_chdhw[0]  # [D, H, W]
    D, H, W = v.shape

    axial = v[D // 2, :, :]                 # (H, W) – OK
    coronal = np.flipud(v[:, H // 2, :])    # (D, W) – flip Z
    sagittal = np.flipud(v[:, :, W // 2])   # (D, H) – flip Z

    return {
        "axial": axial,
        "coronal": coronal,
        "sagittal": sagittal,
    }


def pick_slice(vol_chdhw: np.ndarray, axis: str, index: int) -> np.ndarray:
    """
    vol_chdhw: numpy array [C, D, H, W]
    Returns a 2D slice (H, W) for visualization by:
      - taking channel 0 (or mean across channels if you prefer)
      - slicing along chosen spatial axis
    """
    if vol_chdhw.ndim != 4:
        raise ValueError(f"Expected [C,D,H,W], got {vol_chdhw.shape}")

    # For visualization: use channel 0. (Alternative: vol_chdhw.mean(0))
    v = vol_chdhw[0]  # [D,H,W]

    D, H, W = v.shape
    axis = axis.upper()

    if axis == "D":
        if index < 0:
            index = D // 2
        index = int(np.clip(index, 0, D - 1))
        sl = v[index, :, :]
    elif axis == "H":
        if index < 0:
            index = H // 2
        index = int(np.clip(index, 0, H - 1))
        sl = v[:, index, :]
    elif axis == "W":
        if index < 0:
            index = W // 2
        index = int(np.clip(index, 0, W - 1))
        sl = v[:, :, index]
    else:
        raise ValueError("slice_axis must be one of: D, H, W")

    return sl


def imshow_robust(ax, img, title):
    vmin, vmax = np.percentile(img, [1, 99])
    ax.imshow(img, cmap="gray", vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=10)
    ax.axis("off")



def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, required=True, help="Folder containing pre_surgery/ and 6_months_post_surgery/")
    p.add_argument("--ckpt", type=str, required=True, help="Path to best.pt or last.pt")
    p.add_argument("--base_channels", type=int, default=128, help="UNet base channels (must match training for this ckpt)")
    p.add_argument("--sample_idx", type=int, default=0, help="Dataset index to run inference on")
    p.add_argument("--pre_dirname", type=str, default="pre_surgery")
    p.add_argument("--post_dirname", type=str, default="6_months_post_surgery")
    p.add_argument("--strict", action="store_true", default=True)
    p.add_argument("--loss", type=str, default="l1", choices=["l1", "mse"])
    p.add_argument("--amp", action="store_true", default=True, help="Use autocast on CUDA")
    p.add_argument("--slice_axis", type=str, default="D", choices=["D", "H", "W"])
    p.add_argument("--slice_index", type=int, default=-1, help="-1 means center slice")
    p.add_argument("--save_fig", type=str, default="", help="If set, save figure to this path")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = bool(args.amp and device == "cuda")

    # ---- dataset + transform (must match training) ----
    transform = ToChannelsFirstAndNormalize(nonzero_mask=True)

    ds = PrePostFMRI(
        root_dir=args.data_root,
        pre_dirname=args.pre_dirname,
        post_dirname=args.post_dirname,
        strict=args.strict,
        transform=transform,
        return_paths=True,
    )

    x, y, meta = ds[args.sample_idx]  # x,y: [C,D,H,W]
    in_ch = x.shape[0]
    out_ch = y.shape[0]

    # ---- build model (base_channels must match ckpt) ----
    model = UNet3D(in_channels=in_ch, out_channels=out_ch, base=args.base_channels).to(device)

    ckpt = torch.load(args.ckpt, map_location="cpu")
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state, strict=True)
    model.eval()

    # ---- inference ----
    x_b = x.unsqueeze(0).to(device)  # [1,C,D,H,W]
    y_b = y.unsqueeze(0).to(device)

    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            pred_b = model(x_b)

            if args.loss == "l1":
                loss = l1_loss(pred_b, y_b)
            else:
                loss = mse_loss(pred_b, y_b)

    loss_val = float(loss.detach().cpu())
    print(f"Sample: {args.sample_idx} | id: {meta['id']} | file: {meta['filename']}")
    print(f"Shapes: x={tuple(x.shape)} y={tuple(y.shape)} pred={tuple(pred_b.shape[1:])}")
    print(f"Reconstruction {args.loss.upper()} loss: {loss_val:.6f}")

    # ---- plotting ----
    x_np = x.detach().cpu().numpy()
    y_np = y.detach().cpu().numpy()
    pred_np = pred_b.squeeze(0).detach().cpu().numpy()  # [C,D,H,W]

    # ---- extract slices ----
    x_slices = get_orthogonal_slices(x_np)
    y_slices = get_orthogonal_slices(y_np)
    p_slices = get_orthogonal_slices(pred_np)

    rows = ["axial", "coronal", "sagittal"]
    cols = ["Pre-surgery", "Post-surgery GT", "Prediction"]

    fig, axes = plt.subplots(
        nrows=3, ncols=3, figsize=(10, 10), constrained_layout=True
    )

    for r, view in enumerate(rows):
        imshow_robust(axes[r, 0], x_slices[view], f"{view.capitalize()} | {cols[0]}")
        imshow_robust(axes[r, 1], y_slices[view], f"{view.capitalize()} | {cols[1]}")
        imshow_robust(axes[r, 2], p_slices[view], f"{view.capitalize()} | {cols[2]}")

    fig.suptitle(
        f"ID: {meta['id']}  |  {args.loss.upper()} = {loss_val:.6f}",
        fontsize=12,
    )

    if args.save_fig:
        outp = Path(args.save_fig)
        outp.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(outp, dpi=200, bbox_inches="tight")
        print(f"Saved figure to: {outp}")



if __name__ == "__main__":
    main()
