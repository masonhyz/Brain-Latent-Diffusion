"""
Compare N model checkpoints on the validation set (the shared seed-42 / val_frac
split, 39 subjects) across MAE / MSE / PSNR / SSIM, and draw grouped boxplots
(one panel per metric, one box per model) in a single figure.

Metric regime (default): z-score scale (data_range = target's range). MAE/MSE/PSNR
are BRAIN-MASKED; SSIM is computed over the FULL volume to match training.
Flags: --metric_scale {zscore,rescale01}, --whole_volume, --zero_bg.

Model family is auto-detected from the checkpoint:
  - latent diffusion (ldm_7tcdm3d / cfg* runs): keys {ae, denoiser, args} -> model.generate
  - cdm3d:                                       keys {ema_model, args}    -> model.sample
(Override with --types if needed; tell the maintainer to add a handler for other families.)

All models are evaluated on the SAME 39 val subjects (seed/val_frac below), so the
boxes are directly comparable. This assumes every model shares that split.

All five (or any N) checkpoints may share the same ldm_7tcdm3d family; each is
loaded, evaluated, then freed from the GPU before the next one loads, so memory
stays at one model at a time regardless of N.

Usage (5 ldm_7tcdm3d variants):
  CUDA_VISIBLE_DEVICES=0 python scripts/compare_models_boxplot.py \
      --ckpts runs/cfg1_baseline/stage2_best.pt \
              runs/cfg2_big_denoiser/stage2_best.pt \
              runs/cfg3_rich_latent/stage2_best.pt \
              runs/cfg5_richae_strongcfg/stage2_best.pt \
              runs/cfg5_strong_cfg/stage2_best.pt \
      --labels cfg1 cfg2 cfg3 cfg5_richae cfg5_strong \
      --out outputs/compare_models
"""

import argparse
import csv
import gc
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from torch.utils.data import random_split
from skimage.metrics import structural_similarity

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from moyamoya.dataset import PrePostFMRI
from moyamoya.transform import ToChannelsFirstAndNormalize

METRICS = [("mae", "MAE ↓"), ("mse", "MSE ↓"),
           ("psnr", "PSNR ↑ (dB)"), ("ssim", "SSIM ↑")]


# ── metric ────────────────────────────────────────────────────────────────────
def _rescale01(v):
    lo, hi = float(v.min()), float(v.max())
    return (v - lo) / (hi - lo + 1e-8)


def compute_metric(pred, target, region, bg=None, scale="zscore"):
    """
    MAE/MSE/PSNR over `region`; SSIM over the FULL volume (matches training).
      scale='zscore'    -> raw z-score values; data_range = target's range in `region`
      scale='rescale01' -> min-max rescale each volume to [0,1]; data_range = 1
    """
    p = pred.squeeze().cpu().float().numpy()
    t = target.squeeze().cpu().float().numpy()
    if scale == "rescale01":
        p, t = _rescale01(p), _rescale01(t)
    if bg is not None:
        b = bg.squeeze().cpu().bool().numpy()
        p[b] = 0.0; t[b] = 0.0
    m = region.squeeze().cpu().bool().numpy()
    mp, mt = p[m], t[m]
    mae = float(np.abs(mp - mt).mean())
    mse = float(((mp - mt) ** 2).mean())
    if scale == "rescale01":
        dr = 1.0
    else:
        dr = float(mt.max() - mt.min()); dr = dr if dr > 0 else 1.0
    psnr = float(10.0 * np.log10(dr ** 2 / (mse + 1e-12)))
    # SSIM over the FULL volume (background included), matching training's
    # compute_metrics — the scalar return is the whole-volume mean. data_range
    # is still taken from the brain-masked target (z-score space).
    ssim = float(structural_similarity(t, p, data_range=dr, win_size=7,
                                       channel_axis=None))
    return {"mae": mae, "mse": mse, "psnr": psnr, "ssim": ssim}


# ── model dispatch: returns (model, generate_fn(x_b)->pred, short description) ──
# The model is returned alongside the closure so the caller can free it from the
# GPU between checkpoints — essential when comparing 5+ full-size ldm_7tcdm3d runs.
def load_generate_fn(ckpt_path, mtype, device):
    raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    a = raw["args"]
    if mtype == "auto":
        if "ae" in raw and "denoiser" in raw:
            mtype = "ldm7t"
        elif "ema_model" in raw:
            mtype = "cdm3d"
        else:
            raise ValueError(f"Cannot auto-detect model type for {ckpt_path} "
                             f"(keys={list(raw.keys())}); pass --types")

    if mtype == "ldm7t":
        from moyamoya.models.ldm_7tcdm3d import build_paired_latent_diffusion_7tcdm
        model = build_paired_latent_diffusion_7tcdm(
            z_channels=a["z_channels"], embed_dim=a["embed_dim"], ae_ch=a["ae_ch"],
            ae_ch_mult=tuple(a["ae_ch_mult"]), ae_res_blocks=a["ae_res_blocks"],
            ae_resolution=a["ae_resolution"], kl_weight=a["kl_weight"],
            diff_dim=a["diff_dim"], diff_dim_mults=tuple(a["diff_dim_mults"]),
            init_kernel_size=a["init_kernel_size"], resnet_groups=a["resnet_groups"], T=a["T"],
            schedule=a.get("schedule", "cosine"),
        ).to(device)
        model.ae.load_state_dict(raw["ae"])
        model.denoiser.load_state_dict(raw["denoiser"])
        model.eval()
        ddim, gs = a.get("ddim_steps", 50), a.get("guidance_scale", 3.0)
        fn = lambda x: model.generate(x, ddim_steps=ddim, eta=0.0, guidance_scale=gs)
        return model, fn, f"ldm7t(ddim={ddim},gs={gs})"

    if mtype == "cdm3d":
        from moyamoya.models.cdm3d import build_cdm3d
        model = build_cdm3d(
            dim=a["dim"], dim_mults=tuple(a["dim_mults"]),
            init_kernel_size=a["init_kernel_size"], resnet_groups=a["resnet_groups"],
            timesteps=a["T"], ddim_steps=a["ddim_steps"],
            ssim_weight=a["ssim_weight"], cfg_drop_prob=a["cfg_drop_prob"],
        ).to(device)
        model.load_state_dict(raw["ema_model"])
        model.eval()
        gs = a.get("guidance_scale", 1.0)
        fn = lambda x: model.sample(x, guidance_scale=gs)
        return model, fn, f"cdm3d(gs={gs})"

    raise ValueError(f"Unknown model type '{mtype}'")


def plot_boxplots(results, labels, region_str, out_png, n):
    """Production-style grouped boxplots: one compact panel per metric, one thin box per model."""
    rows = [{"model": lab, "metric": k, "value": v}
            for lab in labels for k, _ in METRICS for v in results[lab][k]]
    df = pd.DataFrame(rows)

    sns.set_theme(style="whitegrid", context="paper", font_scale=0.95)
    pal = sns.color_palette("Set2", n_colors=len(labels))

    fig, axes = plt.subplots(1, len(METRICS), figsize=(2.55 * len(METRICS) + 0.5, 3.7))
    axes = np.atleast_1d(axes)
    for ax, (k, title) in zip(axes, METRICS):
        sub = df[df["metric"] == k]
        sns.boxplot(
            data=sub, x="model", y="value", order=labels, hue="model", hue_order=labels,
            palette=pal, legend=False, ax=ax,
            width=0.5, linewidth=0.9, fliersize=0, saturation=0.9,
            medianprops=dict(color="black", linewidth=1.4),
            whiskerprops=dict(linewidth=0.9), capprops=dict(linewidth=0.9),
        )
        sns.stripplot(data=sub, x="model", y="value", order=labels, ax=ax,
                      color="0.2", size=1.8, alpha=0.35, jitter=0.12)
        ax.set_title(title, fontsize=10.5, weight="bold", pad=5)
        ax.set(xlabel="", ylabel="")
        ax.tick_params(axis="both", labelsize=8)
        for lbl in ax.get_xticklabels():
            lbl.set_rotation(33); lbl.set_ha("right")
        sns.despine(ax=ax, left=True)
    fig.suptitle(f"Validation comparison   ·   n = {n}   ·   {region_str}",
                 fontsize=11.5, weight="bold", y=1.02)
    fig.tight_layout(w_pad=0.6)
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+", required=True, help="Checkpoint paths (one per model)")
    ap.add_argument("--labels", nargs="+", default=None, help="Box labels (default: run-dir names)")
    ap.add_argument("--types", nargs="+", default=None,
                    help="Per-ckpt model type override: auto|ldm7t|cdm3d (default: auto for all)")
    ap.add_argument("--data_root", default="fmri")
    ap.add_argument("--seed",      type=int,   default=42)
    ap.add_argument("--val_frac",  type=float, default=0.15)
    ap.add_argument("--whole_volume", action="store_true")
    ap.add_argument("--zero_bg",      action="store_true")
    ap.add_argument("--metric_scale", choices=["zscore", "rescale01"], default="zscore",
                    help="Metric value scale: zscore (original, data_range=target range) "
                         "or rescale01 ([0,1] min-max, data_range=1)")
    ap.add_argument("--out", default="outputs/compare_models")
    args = ap.parse_args()

    labels = args.labels or [Path(c).resolve().parent.name for c in args.ckpts]
    types  = args.types  or ["auto"] * len(args.ckpts)
    assert len(labels) == len(args.ckpts), "--labels must match --ckpts count"
    assert len(types)  == len(args.ckpts), "--types must match --ckpts count"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)

    # shared validation split (same 39 subjects for every model)
    ds = PrePostFMRI(args.data_root, transform=ToChannelsFirstAndNormalize(nonzero_mask=True),
                     strict=False, return_paths=True)
    n_val = max(1, int(len(ds) * args.val_frac))
    g = torch.Generator().manual_seed(args.seed)
    _, val = random_split(ds, [len(ds) - n_val, n_val], generator=g)
    val_idx = list(val.indices)
    scale_str = "[0,1] rescaled" if args.metric_scale == "rescale01" else "z-score"
    region_str = ("whole-volume" if args.whole_volume else "brain-masked") + \
                 (" + bg->0" if args.zero_bg else "")
    full_str = f"{scale_str}, {region_str}"
    print(f"Val subjects: {len(val_idx)} | {full_str} | models: {len(args.ckpts)}")

    def free_gb():
        if device.type != "cuda":
            return None
        idx = device.index if device.index is not None else torch.cuda.current_device()
        return torch.cuda.mem_get_info(idx)[0] / 1e9

    results = {lab: {k: [] for k, _ in METRICS} for lab in labels}
    for ckpt, lab, mt in zip(args.ckpts, labels, types):
        fg = free_gb()
        if fg is not None:
            print(f"\n[GPU] {fg:.1f} GB free before loading {lab}")
        model, gen_fn, desc = load_generate_fn(ckpt, mt, device)
        print(f"[{lab}] {ckpt}  ({desc})")
        torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
        for j, idx in enumerate(val_idx):
            x, y, meta = ds[idx]
            x_b = x.unsqueeze(0).to(device); y_b = y.unsqueeze(0).to(device)
            with torch.no_grad():
                pred = gen_fn(x_b)
            brain = (x_b != x_b.min()) | (y_b != y_b.min())
            region = torch.ones_like(x_b, dtype=torch.bool) if args.whole_volume else brain
            bg = (~brain) if args.zero_bg else None
            m = compute_metric(pred, y_b, region, bg=bg, scale=args.metric_scale)
            for k, _ in METRICS:
                results[lab][k].append(m[k])
            print(f"  [{j + 1:>2}/{len(val_idx)}] {meta['id']}  "
                  f"MAE={m['mae']:.4f} MSE={m['mse']:.4f} PSNR={m['psnr']:.2f} SSIM={m['ssim']:.4f}")

        # Free this model before the next one loads, so peak GPU memory stays at a
        # single model regardless of how many checkpoints (5+) are compared.
        del model, gen_fn
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # ── summary + CSV ─────────────────────────────────────────────────────────
    csv_path = out_dir / "metrics_long.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["model", "id"] + [k for k, _ in METRICS])
        for lab in labels:
            ids = [ds.pairs[i][0].replace(".nii.gz", "") for i in val_idx]
            for n, sid in enumerate(ids):
                w.writerow([lab, sid] + [f"{results[lab][k][n]:.6f}" for k, _ in METRICS])
    print("\n" + "=" * 64)
    for k, title in METRICS:
        line = "  ".join(f"{lab}={np.mean(results[lab][k]):.4f}" for lab in labels)
        print(f"  {title:14s} mean  {line}")

    out_png = out_dir / "compare_boxplots.png"
    plot_boxplots(results, labels, full_str, out_png, len(val_idx))
    print(f"\nSaved plot: {out_png}\nSaved CSV : {csv_path}")


if __name__ == "__main__":
    main()
