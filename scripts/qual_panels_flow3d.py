"""Per-patient qualitative panels — a 3×5 grid for every subject.

For each patient this renders the three orthogonal views (axial, coronal,
sagittal — rows) through the brain centre against the five columns pre-op /
post-op / prediction (ensemble mean) / |error| / uncertainty (ensemble std),
both maps smoothed to the coherent scale. It streams one figure per subject to
``<out_dir>/panels/<id>.png`` so the whole dataset fits in memory.

The uncertainty column deliberately does **not** share the error's colormap: it
uses a calm, low-chroma cool ramp (``mild_unc``) while the error stays on the hot
magma map, so the two read as different quantities at a glance rather than two
copies of the same fiery gradient.

    # every patient, SDE ensemble (the headline sampler)
    python scripts/qual_panels_flow3d.py \
        --ckpt runs/flow3d_2026-08-02_21-07-41/best_mae.pt --sampler sde

    # just the held-out split, deterministic-ODE ensemble
    python scripts/qual_panels_flow3d.py \
        --ckpt runs/flow3d_2026-08-02_21-07-41/best_mae.pt --val_only --sampler ode
"""

import argparse
import hashlib
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import LinearSegmentedColormap

sys.path.insert(0, str(Path(__file__).parent.parent))

from moyamoya.data import reconstruct_val_split
from moyamoya.dataset import PrePostFMRI
from moyamoya.metrics import foreground_mask
from moyamoya.models.flow3d import gaussian_blur3d, load_flow3d_checkpoint
from moyamoya.transform import ToChannelsFirstAndNormalize
from moyamoya.utils import get_device

C_INK = "#3d3d3a"

# Error stays hot (magma); uncertainty gets its own calm, low-saturation cool
# ramp so the two columns never look like the same gradient. Both mask the
# off-brain background to black to sit cleanly on the figure.
CMAP_ERR = plt.get_cmap("magma").copy()
CMAP_ERR.set_bad("black")
CMAP_UNC = LinearSegmentedColormap.from_list(
    "mild_unc", ["#12212b", "#33546a", "#6893a6", "#a7c6d0", "#e3eef1"])
CMAP_UNC.set_bad("black")


def subject_seed(base: int, sid: str) -> int:
    """Stable per-subject RNG seed (matches eval_flow3d_uncertainty), so a
    subject's ensemble here is identical to the one the analysis reported."""
    h = int(hashlib.sha1(sid.encode()).hexdigest()[:8], 16)
    return (base + h) % (2 ** 31 - 1)


@torch.no_grad()
def sample_ensemble(model, xb, args, steps, seed):
    """K trajectories → (K, D, H, W) fp32 CPU, batched ``chunk`` at a time."""
    g = torch.Generator(device=xb.device).manual_seed(seed)
    outs = []
    for i in range(0, args.n_samples, args.chunk):
        b = min(args.chunk, args.n_samples - i)
        x = xb.repeat(b, 1, 1, 1, 1)
        if args.sampler == "sde":
            p = model.sample_sde(x, steps=steps,
                                 gamma_scale=args.sde_gamma_scale,
                                 guidance_scale=args.guidance_scale, generator=g)
        else:
            p = model.sample(x, steps=steps, solver=args.solver,
                             guidance_scale=args.guidance_scale,
                             init_noise=args.init_noise, generator=g)
        outs.append(p.squeeze(1).float().cpu())
    return torch.cat(outs, 0)


VIEWS = ("axial", "coronal", "sagittal")


def brain_center(fg3: np.ndarray) -> tuple:
    """Voxel centre of mass of the brain → ``(zc, yc, xc)`` ints.

    The single point all three orthogonal planes pass through, so the montage
    is a proper crosshair through the middle of the brain rather than three
    independent slices.
    """
    pts = np.argwhere(fg3)
    if pts.size == 0:
        return tuple(sh // 2 for sh in fg3.shape)
    return tuple(int(round(c)) for c in pts.mean(0))


def view_slice(vol: np.ndarray, view: str, center: tuple) -> np.ndarray:
    """One orthogonal 2-D slice of a ``(Z, Y, X)`` volume, oriented superior-up.

    axial = const Z → (Y, X); coronal = const Y → (Z, X); sagittal = const X →
    (Z, Y). Coronal and sagittal are flipped along Z so the top of the head sits
    at the top of the panel (array row 0 is the inferior-most slice otherwise).
    """
    zc, yc, xc = center
    if view == "axial":
        return vol[zc, :, :]
    if view == "coronal":
        return np.flipud(vol[:, yc, :])
    return np.flipud(vol[:, :, xc])                     # sagittal


def render_patient(sid, x3, y3, mean, err_sm, std_sm, fg3, out_path,
                   sampler_tag):
    """One 3×5 grid for a subject: rows = the three orthogonal views (axial /
    coronal / sagittal) through the brain centre, cols = pre / post / prediction
    / |error| / uncertainty."""
    center = brain_center(fg3)
    cols = ("pre-op", "post-op (truth)", "prediction (mean)",
            "|error| (smoothed)", "uncertainty (smoothed std)")
    row_lab = {"axial": f"axial (z={center[0]})",
               "coronal": f"coronal (y={center[1]})",
               "sagittal": f"sagittal (x={center[2]})"}
    # Stable anatomical window from the in-brain voxels of the three grey panels.
    lo, hi = np.percentile(np.concatenate([x3[fg3], y3[fg3], mean[fg3]]), [1, 99])

    fig, axes = plt.subplots(3, 5, figsize=(12.5, 8.6))
    for r, view in enumerate(VIEWS):
        m = view_slice(fg3, view, center)
        e = view_slice(err_sm, view, center)
        u = view_slice(std_sm, view, center)
        e0, e1 = np.percentile(e[m], [2, 98]) if m.any() else (0, 1)
        u0, u1 = np.percentile(u[m], [2, 98]) if m.any() else (0, 1)
        panels = [(view_slice(x3, view, center), "gray", lo, hi),
                  (view_slice(y3, view, center), "gray", lo, hi),
                  (view_slice(mean, view, center), "gray", lo, hi),
                  (e, CMAP_ERR, e0, e1), (u, CMAP_UNC, u0, u1)]
        for c, (img, cmap, vmin, vmax) in enumerate(panels):
            ax = axes[r, c]
            if c < 3:
                ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
            else:
                ax.imshow(np.ma.masked_where(~m, img), cmap=cmap,
                          vmin=vmin, vmax=max(vmax, vmin + 1e-6))
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            if r == 0:
                ax.set_title(cols[c], fontsize=10, color=C_INK)
        axes[r, 0].set_ylabel(row_lab[view], fontsize=9, color=C_INK)

    mae = float(np.abs(mean - y3)[fg3].mean())
    imae = float(np.abs(x3 - y3)[fg3].mean())
    fig.suptitle(f"{sid}   —   brain MAE {mae:.3f}  (identity {imae:.3f})   "
                 f"[{sampler_tag}]", fontsize=12, color=C_INK)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=170)
    plt.close(fig)
    return mae, imae


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root",   type=str, default="fmri")
    p.add_argument("--ckpt",        type=str, default="runs/flow3d/best_mae.pt")
    p.add_argument("--out_dir",     type=str, default=None)
    p.add_argument("--pre_dirname",  type=str, default="pre_surgery")
    p.add_argument("--post_dirname", type=str, default="6_months_post_surgery")
    p.add_argument("--val_only", action="store_true",
                   help="Restrict to the checkpoint's held-out split "
                        "(default: every patient in the dataset)")
    p.add_argument("--limit", type=int, default=None)
    # ensemble
    p.add_argument("--sampler", type=str, default="sde", choices=["ode", "sde"])
    p.add_argument("--sde_gamma_scale", type=float, default=1.0)
    p.add_argument("--init_noise", type=float, default=0.1)
    p.add_argument("--n_samples", type=int, default=20)
    p.add_argument("--chunk", type=int, default=10)
    p.add_argument("--steps", type=int, default=None,
                   help="Default: 16 for sde, the checkpoint's value for ode")
    p.add_argument("--solver", type=str, default=None,
                   choices=["euler", "heun", "rk4"])
    p.add_argument("--guidance_scale", type=float, default=None)
    p.add_argument("--sample_seed", type=int, default=0)
    p.add_argument("--smooth_sigma", type=float, default=2.0)
    p.add_argument("--use_ema", dest="use_ema", action="store_true")
    p.add_argument("--no-ema",  dest="use_ema", action="store_false")
    p.set_defaults(use_ema=True)
    args = p.parse_args()

    args.device = get_device()
    model, raw = load_flow3d_checkpoint(args.ckpt, device=args.device,
                                        use_ema=args.use_ema)
    ck = raw.get("args", {})
    steps = args.steps or (16 if args.sampler == "sde" else ck.get("steps", 8))
    args.solver = args.solver or ck.get("solver", "heun")
    if args.guidance_scale is None:
        args.guidance_scale = ck.get("guidance_scale", 1.0)
    sampler_tag = (f"SDE γ={args.sde_gamma_scale}·σ², K={args.n_samples}"
                   if args.sampler == "sde"
                   else f"ODE init_noise={args.init_noise}, K={args.n_samples}")

    zero_bg = ck.get("zero_background", False)
    ds = PrePostFMRI(root_dir=args.data_root, pre_dirname=args.pre_dirname,
                     post_dirname=args.post_dirname,
                     transform=ToChannelsFirstAndNormalize(
                         nonzero_mask=True, zero_background=zero_bg),
                     strict=False, return_paths=True)
    if args.val_only:
        seed, vf = ck.get("seed", 42), ck.get("val_frac", 0.15)
        _, val = reconstruct_val_split(ds, vf, seed, n_folds=ck.get("n_folds"),
                                       fold=ck.get("fold"))
        indices, scope = list(val.indices), "held-out validation split"
    else:
        indices, scope = list(range(len(ds))), "every patient"
    if args.limit:
        indices = indices[:args.limit]

    suffix = "_sde" if args.sampler == "sde" else ""
    out_dir = Path(args.out_dir or
                   f"outputs/uncertainty/{Path(args.ckpt).parent.name}{suffix}") / "panels"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Per-patient orthogonal-view panels — {scope}, {len(indices)} subjects")
    print(f"  {sampler_tag}, steps={steps}, smooth σ={args.smooth_sigma}")
    print(f"  → {out_dir}\n")

    blur = lambda v: gaussian_blur3d(v[None, None], args.smooth_sigma)[0, 0]
    for n, i in enumerate(indices, 1):
        x, y, meta = ds[i]
        sid = meta["id"]
        xb = x.unsqueeze(0).to(args.device)
        preds = sample_ensemble(model, xb, args, steps,
                                subject_seed(args.sample_seed, sid))
        mean = preds.mean(0)
        std = preds.std(0, correction=1)
        x3, y3 = x.squeeze(0), y.squeeze(0)
        err_sm = blur((mean - y3).abs()).numpy()
        std_sm = blur(std).numpy()
        fg3 = foreground_mask(x3, y3)
        mae, imae = render_patient(sid, x3.numpy(), y3.numpy(), mean.numpy(),
                                   err_sm, std_sm, fg3, out_dir / f"{sid}.png",
                                   sampler_tag)
        if n % 10 == 0 or n == len(indices):
            print(f"  [{n}/{len(indices)}] {sid}  MAE={mae:.3f} (id {imae:.3f})")

    print(f"\nDone — {len(indices)} panels in {out_dir}")


if __name__ == "__main__":
    main()
