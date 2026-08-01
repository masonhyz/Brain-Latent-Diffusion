"""Orthogonal mid-slice grid rendering, shared across the infer / eval scripts.

Previously the percentile-normalise helper (`_to_np`) and the 3×3
axial/coronal/sagittal figure were copy-pasted into every inference script.
"""

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def percentile_norm(vol, lo_pct: float = 1, hi_pct: float = 99, eps: float = 1e-8) -> np.ndarray:
    """Squeeze to a 3-D numpy volume normalised to [0, 1].

    Range is taken from the (lo_pct, hi_pct) percentiles of the nonzero voxels
    (falling back to min/max when the volume is all-zero).
    """
    if isinstance(vol, torch.Tensor):
        v = vol.squeeze().cpu().float().numpy()
    else:
        v = np.asarray(vol).squeeze()
    lo, hi = np.percentile(v[v != 0], [lo_pct, hi_pct]) if (v != 0).any() else (v.min(), v.max())
    return np.clip((v - lo) / (hi - lo + eps), 0, 1)


def _mid(arr: np.ndarray, axis: int) -> int:
    return arr.shape[axis] // 2


def save_ortho_grid(x_pre, x_real, x_pred, save_path, title: str = "", cmap: str = "hot") -> None:
    """3×3 grid — rows: axial/coronal/sagittal mid-slices, cols: pre / pred / real."""
    pre  = percentile_norm(x_pre)
    real = percentile_norm(x_real)
    pred = percentile_norm(x_pred)

    slices = {
        "Axial":    (pre[_mid(pre, 0)],       real[_mid(real, 0)],       pred[_mid(pred, 0)]),
        "Coronal":  (pre[:, _mid(pre, 1)],    real[:, _mid(real, 1)],    pred[:, _mid(pred, 1)]),
        "Sagittal": (pre[:, :, _mid(pre, 2)], real[:, :, _mid(real, 2)], pred[:, :, _mid(pred, 2)]),
    }

    fig, axes = plt.subplots(3, 3, figsize=(10, 10))
    cols = ["Pre-surgery", "Predicted Post", "Real Post"]
    for row_i, (view, (s_pre, s_real, s_pred)) in enumerate(slices.items()):
        for col_i, (ax, img, col_title) in enumerate(
            zip(axes[row_i], [s_pre, s_pred, s_real], cols)
        ):
            ax.imshow(img, cmap=cmap, origin="lower")
            ax.axis("off")
            if row_i == 0:
                ax.set_title(col_title, fontsize=11)
            if col_i == 0:
                ax.set_ylabel(view, fontsize=10)

    if title:
        fig.suptitle(title, fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {save_path}")
