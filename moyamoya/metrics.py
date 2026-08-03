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


def tissue_mask(x_pre, x_post):
    """Voxels where **both** volumes have signal — the region worth scoring.

    The convention used elsewhere in this repo is the *union*,
    ``(x_pre != 0) | (x_post != 0)``. That is the wrong mask for a comparison.
    Pre- and post-op scans do not cover identical voxels: about 5.4 % of the
    union has tissue in one volume and nothing in the other, and there no
    meaningful pre-vs-post difference exists — the model is being scored on a
    coverage artefact.

    Worse, that region is where the two preprocessing conventions disagree
    wildly. Scoring the identity baseline over the union gives mean |x−y| of
    0.357 with a shifted background (the tissue value happens to sit near the
    −2.4 plateau, so the error partly cancels) and 2.034 with a true zero
    background — from the same voxels. Over the intersection both conventions
    agree exactly, at 0.454. Only the intersection measures the thing we care
    about.

    Args / returns: boolean mask, same shape as the inputs.
    """
    return (x_pre != 0) & (x_post != 0)


def union_mask(x_pre, x_post):
    """Whole-volume mask ``(x_pre != 0) | (x_post != 0)`` — the 7TCDM3D/LDM
    convention. With ``zero_background=False`` normalisation the background is a
    constant plateau (not 0), so this selects every voxel; use it to score a
    model *the same way* the diffusion models here are scored."""
    return (x_pre != 0) | (x_post != 0)


def identity_baseline(pairs, mask_fn=tissue_mask, **kw) -> dict:
    """Metrics of the trivial predictor ``x_post := x_pre``, averaged over ``pairs``.

    This is the number every model on this dataset has to clear, and it is a
    high bar: pre- and post-surgery volumes are the same brain six months apart,
    so copying the input already scores MAE ≈ 0.22 / PSNR ≈ 25.2 / SSIM ≈ 0.84 on
    the seed=42 holdout split. Reporting a model's metrics without it next to
    them is how a model that is *worse than doing nothing* looks like a result.

    Args:
        pairs: any iterable of ``(x_pre, x_post)`` — a Dataset, a Subset, or a
            DataLoader (batched tensors are handled).
        mask_fn: ``(x_pre, x_post) -> mask`` selecting the region to score.
            Defaults to :func:`tissue_mask` (intersection); pass
            :func:`union_mask` to score whole-volume as the 7TCDM3D/LDM models do.
        **kw: forwarded to :func:`compute_metrics` (e.g. ``data_range``).

    Returns:
        Mean over samples of {"mae", "mse", "psnr", "ssim"}.
    """
    acc = {"mae": [], "mse": [], "psnr": [], "ssim": []}
    for x, y in pairs:
        # A DataLoader yields (B,C,D,H,W); a Dataset yields (C,D,H,W).
        batch = [(x[i], y[i]) for i in range(x.shape[0])] if x.ndim == 5 else [(x, y)]
        for xi, yi in batch:
            m = compute_metrics(xi, yi, mask_fn(xi, yi), **kw)
            for k in acc:
                acc[k].append(m[k])
    return {k: float(np.mean(v)) for k, v in acc.items()}
