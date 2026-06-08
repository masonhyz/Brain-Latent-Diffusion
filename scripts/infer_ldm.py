"""
PairedDiffusion inference: generate post-surgery CBF from pre-surgery CBF.

Usage:
    python scripts/infer_ldm.py --ckpt runs/paired_diff/best.pt \\
        --data_root fmri --sample_idx 0 --ddim_steps 50 --out fig.png
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))

from moyamoya.dataset import PrePostFMRI
from moyamoya.transform import ToChannelsFirstAndNormalize
from moyamoya.utils import masked_l1, make_union_mask
from moyamoya.models.ldm import build_paired_diffusion


# ── helpers ──────────────────────────────────────────────────────────────────

def load_model(ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device)
    a    = ckpt["args"]

    model = build_paired_diffusion(
        base=a["diff_base"],
        t_dim=a["t_dim"],
        n_levels=a["n_levels"],
        T=a["T"],
    ).to(device)
    model.denoiser.load_state_dict(ckpt["denoiser"])
    model.eval()
    return model


def _to_np(vol: torch.Tensor) -> np.ndarray:
    """(1, D, H, W) or (D, H, W) → (D, H, W) numpy, percentile-normalised."""
    v = vol.squeeze().cpu().float().numpy()
    lo, hi = np.percentile(v[v != 0], [1, 99]) if (v != 0).any() else (v.min(), v.max())
    return np.clip((v - lo) / (hi - lo + 1e-8), 0, 1)


def _mid(arr: np.ndarray, axis: int) -> int:
    return arr.shape[axis] // 2


def visualize(x_pre, x_post_real, x_post_pred, loss_val: float, save_path: str):
    """
    3-column figure: pre-surgery | predicted post | real post
    3 rows: axial | coronal | sagittal
    """
    pre  = _to_np(x_pre)
    real = _to_np(x_post_real)
    pred = _to_np(x_post_pred)

    slices = {
        "Axial":    (pre[_mid(pre,0)],     real[_mid(real,0)],     pred[_mid(pred,0)]),
        "Coronal":  (pre[:,_mid(pre,1)],   real[:,_mid(real,1)],   pred[:,_mid(pred,1)]),
        "Sagittal": (pre[:,:,_mid(pre,2)], real[:,:,_mid(real,2)], pred[:,:,_mid(pred,2)]),
    }

    fig, axes = plt.subplots(3, 3, figsize=(10, 10))
    cols = ["Pre-surgery", "Predicted Post", "Real Post"]
    for row_i, (view, (s_pre, s_real, s_pred)) in enumerate(slices.items()):
        for col_i, (ax, img, title) in enumerate(
            zip(axes[row_i], [s_pre, s_pred, s_real], cols)
        ):
            ax.imshow(img, cmap="hot", origin="lower")
            ax.axis("off")
            if row_i == 0:
                ax.set_title(title, fontsize=11)
            if col_i == 0:
                ax.set_ylabel(view, fontsize=10)

    fig.suptitle(f"PairedDiffusion prediction  |  masked-L1 = {loss_val:.4f}", fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {save_path}")


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt",       required=True,  help="Path to best.pt checkpoint")
    p.add_argument("--data_root",  default="fmri")
    p.add_argument("--sample_idx", type=int,   default=0)
    p.add_argument("--ddim_steps", type=int,   default=50)
    p.add_argument("--eta",            type=float, default=0.0,
                   help="DDIM stochasticity (0=deterministic, 1=DDPM)")
    p.add_argument("--guidance_scale", type=float, default=3.0,
                   help="CFG guidance scale (1.0 = no guidance, requires model trained with cfg_drop_prob > 0)")
    p.add_argument("--out",            default="ldm_prediction.png")
    p.add_argument("--n_samples",  type=int,   default=1,
                   help="Number of samples to average (>1 explores stochasticity)")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = load_model(args.ckpt, device)

    tfm = ToChannelsFirstAndNormalize(nonzero_mask=True)
    ds  = PrePostFMRI(root_dir=args.data_root, transform=tfm, strict=False)
    x, y = ds[args.sample_idx]
    x = x.unsqueeze(0).to(device)
    y = y.unsqueeze(0).to(device)
    print(f"Sample {args.sample_idx}: pre={tuple(x.shape)}  post={tuple(y.shape)}")

    with torch.no_grad():
        if args.n_samples == 1:
            y_pred = model.generate(x, ddim_steps=args.ddim_steps, eta=args.eta,
                                    guidance_scale=args.guidance_scale)
        else:
            preds  = [model.generate(x, ddim_steps=args.ddim_steps, eta=args.eta,
                                     guidance_scale=args.guidance_scale)
                      for _ in range(args.n_samples)]
            y_pred = torch.stack(preds, dim=0).mean(dim=0)

    mask    = make_union_mask(x, y)
    loss    = masked_l1(y_pred, y, mask).item()
    id_loss = masked_l1(x, y, mask).item()
    print(f"Masked L1 (predicted vs real post): {loss:.4f}")
    print(f"Identity  (pre      vs real post):  {id_loss:.4f}")

    visualize(x[0], y[0], y_pred[0], loss, args.out)


if __name__ == "__main__":
    main()
