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
from scipy.ndimage import gaussian_filter
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


def foreground_mask(*vols) -> np.ndarray:
    """Brain (non-background) voxels — the *union* across the given volumes.

    Both preprocessing conventions here put the background at a single constant
    value: the minimum plateau with ``zero_background=False`` (the whole volume is
    z-scored, so every out-of-brain voxel shares one value) and exact 0 with it on.
    Either way the background is by far the most frequent *exact* value, so it is
    detected per volume as the modal value and everything else is foreground. This
    is robust to both conventions, unlike ``x != 0`` (which selects the whole
    volume when the background is a nonzero plateau).
    """
    m = None
    for v in vols:
        a = _as_np(v).astype(np.float32)
        vals, counts = np.unique(a, return_counts=True)
        bg = vals[int(counts.argmax())]
        fg = a != bg
        m = fg if m is None else (m | fg)
    return m


def change_mask(x_pre, x_post, frac: float = 0.05, min_abs: float = 0.0,
                brain=None) -> np.ndarray:
    """The **region of change**: the top-``frac`` most-changed brain voxels.

    Surgery alters the scan in a small, localised region; everywhere else
    x_post ≈ x_pre and the identity/copy baseline is already perfect. That is
    exactly why the whole-volume metrics are blind to whether a model learned the
    edits — the changed region is a few percent of voxels, so nailing it barely
    moves a mean taken over the whole volume (or, worse, over the ~71%-background
    union). This ROI is where a model can actually beat identity, and the only
    place worth scoring to answer "did it learn the substantial changes?".

    Defined per subject as the brain voxels (see :func:`foreground_mask`) whose
    ``|x_post − x_pre|`` is in the top ``frac`` fraction, optionally floored at an
    absolute ``min_abs`` (z-score units). ``brain`` overrides the auto foreground.
    """
    xp = _as_np(x_pre).astype(np.float32)
    xq = _as_np(x_post).astype(np.float32)
    diff = np.abs(xq - xp)
    brain = foreground_mask(x_pre, x_post) if brain is None else _as_np(brain).astype(bool)
    d = diff[brain]
    if d.size == 0:
        return np.zeros(diff.shape, dtype=bool)
    thr = max(float(np.quantile(d, 1.0 - frac)), float(min_abs))
    return brain & (diff >= thr)


def coherent_change_split(x_pre, x_post, sigma: float = 2.0):
    """Split the pre→post edit ``Δ = x_post − x_pre`` into a low-frequency
    **coherent** component and a high-frequency residual.

    ``Δ_coh = gaussian(Δ, σ)`` is the smooth, spatially-coherent part of the edit
    — the part that is reproducible across acquisitions and that a model can hope
    to predict from ``x_pre``. The residual ``Δ_hf = Δ − Δ_coh`` is high-frequency
    speckle, and on this dataset it is **not biology**: pre- and post-op scans are
    separate acquisitions ~6 months apart, warped to a common space, so ``Δ_hf`` is
    dominated by residual mis-registration and acquisition noise. Measured across
    all 235 pairs, ~60 % of the change *energy* inside the top-5 % change ROI lives
    in ``Δ_hf`` — it is unpredictable from ``x_pre`` by construction, so scoring a
    model against it just penalises the model for not hallucinating noise. σ=2
    voxels is the scale at which an oracle smooth edit recovers ~62 % of the ROI
    error (see the change-signal analysis / README §3).

    Returns ``(Δ_coh, Δ_hf)`` as float32 arrays with the (squeezed) input shape.
    """
    xp = _as_np(x_pre).astype(np.float32)
    xq = _as_np(x_post).astype(np.float32)
    delta = xq - xp
    delta_coh = gaussian_filter(delta, sigma=sigma).astype(np.float32)
    return delta_coh, (delta - delta_coh).astype(np.float32)


def edge_enrichment(x_pre, roi, brain=None, edge_frac: float = 0.20) -> float:
    """How much the change ``roi`` over-represents ``x_pre``'s structural edges.

    Defines edges as the top ``edge_frac`` of brain voxels by ``|∇x_pre|``, so a
    region drawn at random overlaps them at rate ``edge_frac`` by construction.
    The returned ratio ``mean(edge[roi]) / edge_frac`` is therefore 1.0 at chance;
    values well above 1 are the **mis-registration signature** — the largest
    pre→post differences pile up on the moving anatomical edges, where a sub-voxel
    warp error produces a large ``|Δ|``. Measured ~2.2× on this dataset.
    """
    a = _as_np(x_pre).astype(np.float32)
    grad = np.sqrt(sum(g ** 2 for g in np.gradient(a)))
    b = foreground_mask(x_pre) if brain is None else _as_np(brain).astype(bool)
    r = _as_np(roi).astype(bool)
    gb = grad[b]
    if gb.size == 0 or r.sum() == 0 or edge_frac <= 0:
        return float("nan")
    thr = float(np.quantile(gb, 1.0 - edge_frac))
    edges = b & (grad >= thr)
    return float(edges[r].mean()) / edge_frac


def change_region_report(pred, x_pre, x_post, frac: float = 0.05,
                         data_range: float | None = None, coherent: bool = True,
                         sigma: float = 2.0, edge_frac: float = 0.20) -> dict:
    """Score ``pred`` against ``x_post`` **inside the region of change**, next to
    what the identity baseline (copy x_pre) scores in the same ROI.

    Returns model MAE/PSNR/SSIM over the change ROI, the identity baseline's ROI
    MAE (= the mean edit magnitude the model has to explain — large by
    construction), and ``change_mae_improvement`` = identity − model. A positive
    improvement is the thing the whole-volume metrics cannot show: the model
    genuinely reduced error where the surgery actually changed the scan. The ROI
    is :func:`change_mask` (top ``frac`` most-changed brain voxels).

    **The change ROI is ~60 % registration/acquisition noise**, so ``change_mae``
    scores the model partly against noise it should not try to predict (see
    :func:`coherent_change_split`). Pass ``coherent=True`` to *also* get the honest,
    noise-aware **coherent** family — the same four metrics computed on the
    registration-noise-stripped (σ=``sigma`` low-pass) change, plus two diagnostics:

    * ``coherent_mae`` / ``coherent_mse`` / ``coherent_psnr`` / ``coherent_ssim`` —
      model error on the coherent change *image* ``x_pre + gaussian(pred − x_pre)``
      against the coherent target ``x_pre + gaussian(x_post − x_pre)`` in the ROI.
      Smoothing both sides means correct high-freq content is neither rewarded nor
      punished (identity → 0 improvement, perfect → 0 error).
    * ``identity_coherent_mae`` / ``coherent_mae_improvement`` — the identity bar on
      the coherent change and the model's signed win over it. **The paper number.**
    * ``coherent_frac`` — fraction of the ROI change *energy* that is coherent
      (~0.4 here; ``1 − coherent_frac`` is the registration/noise floor).
    * ``edge_enrichment`` — how much the ROI over-represents ``x_pre`` edges vs
      chance (:func:`edge_enrichment`); >1 is the mis-registration signature (~2.2×).

    ``coherent=True`` (the default) reports both families. Pass ``coherent=False``
    for a lean report of only the six raw ``change_*`` keys.
    """
    roi = change_mask(x_pre, x_post, frac=frac)
    base_keys = ("change_mae", "change_psnr", "change_ssim", "identity_change_mae",
                 "change_mae_improvement", "change_roi_frac")
    coh_keys = ("coherent_mae", "coherent_mse", "coherent_psnr", "coherent_ssim",
                "identity_coherent_mae", "coherent_mae_improvement",
                "coherent_frac", "edge_enrichment")
    keys = base_keys + (coh_keys if coherent else ())
    if roi.sum() == 0:
        return {k: (0.0 if k == "change_roi_frac" else float("nan")) for k in keys}

    model = compute_metrics(pred,  x_post, roi, data_range=data_range)
    ident = compute_metrics(x_pre, x_post, roi, data_range=data_range)
    out = {
        "change_mae":  model["mae"],
        "change_psnr": model["psnr"],
        "change_ssim": model["ssim"],
        "identity_change_mae": ident["mae"],
        # >0 ⇒ the model beats copy-x_pre where it actually matters.
        "change_mae_improvement": ident["mae"] - model["mae"],
        "change_roi_frac": float(roi.mean()),
    }
    if not coherent:
        return out

    # ── coherent (registration-noise-stripped) family ────────────────────────
    r = _as_np(roi).astype(bool)
    xp = _as_np(x_pre).astype(np.float32)
    delta_coh, delta_hf = coherent_change_split(x_pre, x_post, sigma=sigma)
    e_coh = float((delta_coh[r] ** 2).sum())
    e_tot = e_coh + float((delta_hf[r] ** 2).sum())
    # Coherent change *images* (x_pre + smoothed edit), so the family parallels the
    # whole-volume metrics and the x_pre baseline is exact (gaussian(0)=0).
    coh_tgt  = xp + delta_coh
    coh_pred = xp + gaussian_filter(_as_np(pred).astype(np.float32) - xp,
                                    sigma=sigma).astype(np.float32)
    m  = compute_metrics(coh_pred, coh_tgt, roi, data_range=data_range)
    bi = compute_metrics(xp,       coh_tgt, roi, data_range=data_range)  # identity
    out.update({
        "coherent_mae":  m["mae"],
        "coherent_mse":  m["mse"],
        "coherent_psnr": m["psnr"],
        "coherent_ssim": m["ssim"],
        "identity_coherent_mae": bi["mae"],
        # >0 ⇒ the model recovered coherent edit that copying x_pre did not.
        "coherent_mae_improvement": bi["mae"] - m["mae"],
        "coherent_frac": (e_coh / e_tot) if e_tot > 0 else float("nan"),
        "edge_enrichment": edge_enrichment(x_pre, roi, edge_frac=edge_frac),
    })
    return out


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
