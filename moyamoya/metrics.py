"""Shared image-quality metrics for the diffusion / prediction models.

Previously copy-pasted (MAE / MSE / PSNR / SSIM, brain-masked, z-score space)
across train_cdm3d, eval_cdm3d, train_ldm_7tcdm3d, and eval_ldm_7tcdm3d.
"""

import numpy as np
import torch
from skimage.metrics import structural_similarity


def compute_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    sanitize_pred: bool = False,
) -> dict:
    """MAE, MSE, PSNR, SSIM between pred and target within the brain ``mask``.

    Args:
        pred, target: tensors, squeezed to a single 3-D volume.
        mask: boolean/soft mask; PSNR/MAE/MSE are computed over ``mask`` voxels,
            SSIM over the full volume with ``data_range`` from the masked target.
        sanitize_pred: replace NaN/Inf in ``pred`` with 0 before scoring (used
            during training where sampled predictions can be unstable).
    """
    p = pred.squeeze().cpu().float().numpy()
    if sanitize_pred:
        p = np.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0)
    t = target.squeeze().cpu().float().numpy()
    m = mask.squeeze().cpu().bool().numpy()

    mp, mt = p[m], t[m]
    mae = float(np.abs(mp - mt).mean())
    mse = float(((mp - mt) ** 2).mean())
    data_range = float(mt.max() - mt.min()) if mt.max() > mt.min() else 1.0
    psnr = float(10 * np.log10(data_range ** 2 / (mse + 1e-12)))
    ssim_val = float(structural_similarity(
        t, p, data_range=data_range, win_size=7, channel_axis=None,
    ))
    return {"mae": mae, "mse": mse, "psnr": psnr, "ssim": ssim_val}
