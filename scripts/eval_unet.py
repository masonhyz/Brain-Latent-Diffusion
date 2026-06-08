"""
eval_unet.py

Evaluate a UNet3D checkpoint on all samples in the PrePostFMRI dataset.
For each sample, saves a 3×3 grid (axial / coronal / sagittal) ×
(pre-surgery | post-surgery GT | prediction).

Also writes a CSV with per-sample masked-L1 and identity-baseline metrics.

Usage:
  python scripts/eval_unet.py \\
    --data_root fmri \\
    --ckpt runs/unet_prepost_bc128/best.pt \\
    --base_channels 128 \\
    --out_dir outputs/eval_unet_bc128

The script auto-detects channels from the first dataset sample, so
--base_channels is the only model hyperparameter you need to supply.
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
from moyamoya.models.unet import UNet3D
from moyamoya.transform import ToChannelsFirstAndNormalize
from moyamoya.utils import make_union_mask, masked_l1


# ---------------------------------------------------------------------------
# Slice helpers
# ---------------------------------------------------------------------------

def _get_orthogonal_slices(vol: np.ndarray) -> dict:
    """vol: [C, D, H, W] → dict of three 2-D slices (channel 0, center planes)."""
    v = vol[0]          # [D, H, W]
    D, H, W = v.shape
    return {
        "axial":    v[D // 2, :, :],
        "coronal":  np.flipud(v[:, H // 2, :]),
        "sagittal": np.flipud(v[:, :, W // 2]),
    }


def _imshow(ax, img: np.ndarray, title: str) -> None:
    vmin, vmax = np.percentile(img, [1, 99])
    ax.imshow(img, cmap="gray", vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.set_title(title, fontsize=9)
    ax.axis("off")


# ---------------------------------------------------------------------------
# Per-sample inference + grid
# ---------------------------------------------------------------------------

@torch.no_grad()
def infer_sample(model, x: torch.Tensor, y: torch.Tensor, device: str, amp: bool):
    """
    x, y : [C, D, H, W] CPU tensors
    Returns pred [C, D, H, W] CPU, masked_l1 scalar, identity_masked_l1 scalar.
    """
    x_b = x.unsqueeze(0).to(device)   # [1, C, D, H, W]
    y_b = y.unsqueeze(0).to(device)
    mask = make_union_mask(x_b, y_b)

    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
        delta = model(x_b)
        pred_b = x_b + delta           # delta prediction (matches training)
        loss = masked_l1(pred_b, y_b, mask)
        id_loss = masked_l1(x_b, y_b, mask)

    pred = pred_b.squeeze(0).cpu()
    return pred, float(loss), float(id_loss)


def save_grid(
    x_np: np.ndarray,
    y_np: np.ndarray,
    pred_np: np.ndarray,
    sample_id: str,
    loss: float,
    id_loss: float,
    save_path: Path,
) -> None:
    views = ["axial", "coronal", "sagittal"]
    cols = ["Pre-surgery", "Post-surgery GT", "Prediction"]

    xs = _get_orthogonal_slices(x_np)
    ys = _get_orthogonal_slices(y_np)
    ps = _get_orthogonal_slices(pred_np)

    fig, axes = plt.subplots(3, 3, figsize=(10, 10), constrained_layout=True)

    for r, view in enumerate(views):
        _imshow(axes[r, 0], xs[view], f"{view.capitalize()} | {cols[0]}")
        _imshow(axes[r, 1], ys[view], f"{view.capitalize()} | {cols[1]}")
        _imshow(axes[r, 2], ps[view], f"{view.capitalize()} | {cols[2]}")

    fig.suptitle(
        f"ID: {sample_id}  |  masked-L1 pred={loss:.4f}  id={id_loss:.4f}",
        fontsize=11,
    )
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root",     type=str, default="fmri")
    p.add_argument("--ckpt",          type=str, default="runs/unet_prepost_bc128/best.pt")
    p.add_argument("--base_channels", type=int, default=128)
    p.add_argument("--out_dir",       type=str, default="outputs/eval_unet_bc128")
    p.add_argument("--pre_dirname",   type=str, default="pre_surgery")
    p.add_argument("--post_dirname",  type=str, default="6_months_post_surgery")
    p.add_argument("--amp",           action="store_true", default=True)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    amp = bool(args.amp and device == "cuda")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- dataset ----
    transform = ToChannelsFirstAndNormalize(nonzero_mask=True)
    ds = PrePostFMRI(
        root_dir=args.data_root,
        pre_dirname=args.pre_dirname,
        post_dirname=args.post_dirname,
        strict=True,
        transform=transform,
        return_paths=True,
    )
    print(f"Dataset: {len(ds)} samples")

    # ---- model ----
    x0, y0, _ = ds[0]
    in_ch, out_ch = x0.shape[0], y0.shape[0]

    model = UNet3D(in_channels=in_ch, out_channels=out_ch, base=args.base_channels).to(device)
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state, strict=True)
    model.eval()

    meta_epoch = ckpt.get("epoch", "?")
    meta_val = ckpt.get("best_val", float("nan"))
    print(f"Loaded: {args.ckpt}  (epoch={meta_epoch}, best_val={meta_val:.6f})")
    print(f"Output: {out_dir}\n")

    # ---- evaluate all samples ----
    rows = []
    for idx in range(len(ds)):
        x, y, meta = ds[idx]
        sid = meta["id"]

        pred, loss, id_loss = infer_sample(model, x, y, device, amp)

        grid_path = out_dir / f"{sid}.png"
        save_grid(
            x.numpy(), y.numpy(), pred.numpy(),
            sid, loss, id_loss, grid_path,
        )

        rows.append({"id": sid, "masked_l1": loss, "identity_l1": id_loss})
        print(f"[{idx+1:>3}/{len(ds)}] {sid}  pred={loss:.4f}  id={id_loss:.4f}  -> {grid_path.name}")

    # ---- summary CSV ----
    csv_path = out_dir / "metrics.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "masked_l1", "identity_l1"])
        writer.writeheader()
        writer.writerows(rows)

    losses = [r["masked_l1"] for r in rows]
    id_losses = [r["identity_l1"] for r in rows]
    print(f"\n{'='*60}")
    print(f"Samples:           {len(rows)}")
    print(f"Mean masked-L1:    pred={np.mean(losses):.5f}  id={np.mean(id_losses):.5f}")
    print(f"Median masked-L1:  pred={np.median(losses):.5f}  id={np.median(id_losses):.5f}")
    print(f"Metrics saved:     {csv_path}")


if __name__ == "__main__":
    main()
