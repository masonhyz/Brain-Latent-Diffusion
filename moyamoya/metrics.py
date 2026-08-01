"""Shared image-quality metrics for the diffusion / prediction models.

Single source of truth for MAE / MSE / PSNR / SSIM, so training, eval, and all
comparison scripts report numbers computed exactly the same way.

Standard (see README → Metrics):
  * All four metrics are **brain-masked** — scored only over ``mask`` voxels.
    - MAE / MSE: mean over masked voxels.
    - PSNR: from the masked MSE.
    - SSIM: the SSIM *map* is computed on the volume, then averaged over the
      mask (``mask_ssim=True``, the default) so the trivially-perfect zero
      background does not inflate it. Set ``mask_ssim=False`` for the legacy
      whole-volume SSIM.
  * z-score space: values are scored as given. ``data_range`` (shared by PSNR
    and SSIM) defaults to the masked target's (max − min). Pass a fixed value
    (e.g. 1.0 after rescaling both volumes to [0, 1]) for cross-study reporting.
"""

import numpy as np
import torch
from skimage.metrics import structural_similarity


def _as_np(a) -> np.ndarray:
    """Squeeze a torch tensor or array-like to a numpy array."""
    if isinstance(a, torch.Tensor):
        a = a.detach().cpu().numpy()
    return np.asarray(a).squeeze()


def compute_metrics(
    pred,
    target,
    mask,
    sanitize_pred: bool = True,
    data_range: float | None = None,
    mask_ssim: bool = True,
) -> dict:
    """MAE, MSE, PSNR, SSIM between ``pred`` and ``target`` within ``mask``.

    Args:
        pred, target: 3-D volume (torch tensor or numpy array), same shape.
        mask: boolean/soft mask selecting the region to score (e.g. brain).
        sanitize_pred: replace NaN/Inf in ``pred`` with 0 before scoring (sampled
            predictions can be unstable early in training).
        data_range: value range for PSNR/SSIM. ``None`` → masked target max−min.
        mask_ssim: average the SSIM map over ``mask`` (default). ``False`` scores
            SSIM over the whole volume (legacy behaviour).

    Returns:
        {"mae", "mse", "psnr", "ssim"} as floats.
    """
    p = _as_np(pred).astype(np.float32)
    if sanitize_pred:
        p = np.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0)
    t = _as_np(target).astype(np.float32)
    m = _as_np(mask).astype(bool)

    mp, mt = p[m], t[m]
    mae = float(np.abs(mp - mt).mean())
    mse = float(((mp - mt) ** 2).mean())

    if data_range is None:
        data_range = float(mt.max() - mt.min()) if mt.max() > mt.min() else 1.0
    psnr = float(10 * np.log10(data_range ** 2 / (mse + 1e-12)))

    if mask_ssim:
        # SSIM map over the volume, then averaged over brain voxels only.
        _, smap = structural_similarity(
            t, p, data_range=data_range, win_size=7, channel_axis=None, full=True,
        )
        ssim_val = float(smap[m].mean())
    else:
        ssim_val = float(structural_similarity(
            t, p, data_range=data_range, win_size=7, channel_axis=None,
        ))

    return {"mae": mae, "mse": mse, "psnr": psnr, "ssim": ssim_val}
