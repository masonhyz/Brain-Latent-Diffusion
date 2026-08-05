"""
Flow3D training — conditional flow matching for paired pre→post CBF prediction.

Single stage, image space, no autoencoder. See moyamoya/models/flow3d.py for
why the default transports x_pre → x_post rather than noise → x_post.

    # default: bridge from the pre-op volume
    python scripts/train_flow3d.py --out_dir runs/flow3d

    # RECOMMENDED (2026-08-05 change-detection rework): factorise WHERE (a
    # supervised change-localisation gate) from WHAT (the residual), and regress
    # the change at the scales where it is predictable. See the change-signal
    # diagnosis — the top-few-% change is ~half unpredictable registration noise,
    # so a plain/re-weighted MSE either collapses to identity or blurs.
    python scripts/train_flow3d.py --gated --det_weight 0.3 --ms_weights 1 1 1 \
        --change_weight 0 --edge_downweight 0.3 --wandb_group flow3d_change_detect

    # ablation: standard CFM from Gaussian noise, same U-Net, for a like-for-like
    # comparison against the diffusion models
    python scripts/train_flow3d.py --source noise --sigma 0 --cfg_drop_prob 0.15 \
        --guidance_scale 3.0 --out_dir runs/flow3d_noise

Every run prints, and plots, the identity baseline (predict x_post := x_pre) on
its own validation split. On this dataset that baseline beats every diffusion
model in the repo, so it is the number that matters. The gated model additionally
reports gate AUC / corr — whether it learned WHERE the scan actually changes.
"""

import argparse
import random
import sys
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from moyamoya.data import build_loaders
from moyamoya.metrics import (
    change_region_report, compute_metrics, foreground_mask, identity_baseline,
    union_mask,
)
from moyamoya.models.flow3d import (
    EMA, build_discriminator3d, build_flow3d, coherent_change_target,
    d_hinge_loss, g_hinge_loss,
)
from moyamoya.runlog import (
    MetricsCSV, init_wandb, install_run_logger, plot_progression, save_grid,
    save_hparams, wandb_finish, wandb_log, wandb_log_image,
)
from moyamoya.utils import get_device, resolve_seed, seed_everything
from moyamoya.viz import percentile_norm as _to_np


METRIC_HIGHER_BETTER = {"mae": False, "mse": False, "psnr": True, "ssim": True}


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root",   type=str,   default="fmri")
    p.add_argument("--out_dir",     type=str,   default=None,
                   help="Run directory. Default: runs/flow3d_<timestamp>/")
    # ── data ─────────────────────────────────────────────────────────────────
    p.add_argument("--val_frac",    type=float, default=0.15,
                   help="Holdout validation fraction (ignored when --fold is set)")
    p.add_argument("--num_workers", type=int,   default=4)
    p.add_argument("--seed",        type=int,   default=None,
                   help="RNG seed for split + init. Default: draw one and log it.")
    p.add_argument("--n_folds",     type=int,   default=7,
                   help="Number of CV folds (used only when --fold is set)")
    p.add_argument("--fold",        type=int,   default=None,
                   help="Held-out fold index 0..n_folds-1. Unset = --val_frac holdout.")
    p.add_argument("--zero_background",    dest="zero_background", action="store_true",
                   help="Restore exact zeros outside the acquired volume after "
                        "z-scoring. OFF by default to match the 7TCDM3D / LDM "
                        "models, which z-score the whole volume (background lands "
                        "at ~-2.4) and score every voxel. Turning it on gives a "
                        "real brain mask but is NOT how the other models here are "
                        "evaluated, so the numbers stop being comparable.")
    p.add_argument("--no-zero-background", dest="zero_background", action="store_false")
    p.set_defaults(zero_background=False)
    # ── augmentation (paired; see moyamoya/transform.py) ─────────────────────
    p.add_argument("--augment",    dest="augment", action="store_true")
    p.add_argument("--no-augment", dest="augment", action="store_false")
    p.set_defaults(augment=True)
    p.add_argument("--aug_flip_p",     type=float, default=0.5)
    p.add_argument("--aug_rotate_deg", type=float, default=10.0)
    p.add_argument("--aug_noise_std",  type=float, default=0.0,
                   help="Input-only Gaussian noise std; the target stays clean. "
                        "Defaults to 0 here (the diffusion models use 0.1) because "
                        "with --source pre the input is not only the conditioning, "
                        "it is the ODE's starting point — see the warning in main().")
    # ── flow matching ────────────────────────────────────────────────────────
    p.add_argument("--source", type=str, default="pre", choices=["pre", "noise"],
                   help="Source distribution of the flow. 'pre' transports the "
                        "pre-op volume to the post-op volume (bridge); 'noise' is "
                        "standard CFM from N(0,I).")
    p.add_argument("--condition", type=str, default="concat", choices=["none", "concat"],
                   help="How x_pre reaches the network. 'concat' (default, and "
                        "effectively required): x_pre is channel-concatenated so "
                        "the network is conditioned on it — without this the loss "
                        "barely descends. 'none' is an ablation that sees x_t only.")
    p.add_argument("--sigma",  type=float, default=0.3,
                   help="Constant noise scale in x_t=(1-t)*x0+t*x1+sigma*eps "
                        "(official ConditionalFlowMatcher path). For the bridge it "
                        "must be > 0: sigma=0 makes x_post recoverable as "
                        "(x_t-x_pre)/t and sampling collapses. Smooths the field.")
    p.add_argument("--t_dist", type=str, default="uniform",
                   choices=["uniform", "logit_normal"],
                   help="Training-time distribution over t. logit_normal is the "
                        "SD3 schedule (concentrates on mid-path).")
    p.add_argument("--logit_mean", type=float, default=0.0)
    p.add_argument("--logit_std",  type=float, default=1.0)
    # ── change-aware training (the identity-collapse fix) ─────────────────────
    p.add_argument("--change_weight", type=float, default=5.0,
                   help="gamma in the per-voxel loss weight w=1+gamma*|dchange|/scale "
                        "(renormalised to mean 1 per sample). Up-weights the voxels "
                        "that actually differ between pre and post so the sparse "
                        "surgical edits are not averaged away by the near-identity "
                        "majority — the cause of the copy-x_pre collapse. 0 = plain "
                        "MSE (ablation). Only applied for --source pre. NOTE: the "
                        "coherent-change approach (--gated + --ms_weights) sets this "
                        "to 0 — up-weighting the raw |Δ| tail chases registration "
                        "noise (see the change-detection block below).")
    p.add_argument("--change_sampler_beta", type=float, default=0.0,
                   help="Oversample big-change subjects with a WeightedRandomSampler, "
                        "weight ∝ 1+beta*g/median(g) with g the subject's global "
                        "change. 0 = uniform. Keep modest (<=1) with ~200 subjects — "
                        "the per-voxel --change_weight is the safer primary lever; "
                        "this is the complementary coarse one (your 'focus on the "
                        "pairs that differ' idea).")
    p.add_argument("--change_sampler_coherent", dest="change_sampler_coherent",
                   action="store_true",
                   help="Weight the change-emphasis sampler by *coherent* (Gaussian-"
                        "smoothed) change, not raw |Δ|. Raw |Δ| is inflated by "
                        "misregistration speckle, so oversampling on it partly "
                        "oversamples the noisiest subjects; the coherent measure "
                        "picks the subjects with big *real* edits. Only matters when "
                        "--change_sampler_beta > 0.")
    p.set_defaults(change_sampler_coherent=False)
    # ── change-detection / coherent-change training (the 2026-08-05 rework) ────
    # Diagnosis: the top-few-% change is ~half unpredictable registration/acquisition
    # noise; the learnable part is the coherent, low-frequency, subject-specific edit.
    # So: factorise WHERE (a supervised detection gate) from WHAT (a residual), and
    # regress the change at the scales where it is predictable. Recommended run:
    #   python scripts/train_flow3d.py --gated --det_weight 0.3 --ms_weights 1 1 1 \
    #       --change_weight 0 --edge_downweight 0.3 --wandb_group flow3d_change_detect
    p.add_argument("--gated", dest="gated", action="store_true",
                   help="Factorise the velocity as gate·residual with a 2-channel "
                        "backbone (GatedVelocityNet). The gate is a supervisable "
                        "change-localisation map — the 'learn WHERE to edit' head. "
                        "Pair with --det_weight; on its own (det_weight 0) the gate "
                        "is unsupervised and this is just an ablation.")
    p.set_defaults(gated=False)
    p.add_argument("--det_weight", type=float, default=0.0,
                   help="Weight of the change-detection loss: soft-Dice between the "
                        "gate and the coherent-change target (smoothed-|Δ| region). "
                        "The imbalance-robust 'learn WHERE' term the re-weighted MSE "
                        "could not be. Requires --gated and --source pre. 0 = off. "
                        "Try 0.3 (comparable in magnitude to the velocity regression "
                        "early on; lower it if the residual stops learning magnitude, "
                        "raise it if the gate stays diffuse).")
    p.add_argument("--det_sigma", type=float, default=2.0,
                   help="Gaussian sigma (voxels) defining the coherent-change region "
                        "the gate is trained to detect. Larger = smoother/coarser.")
    p.add_argument("--ms_weights", type=float, nargs="+", default=None,
                   help="Per-scale weights for the multi-scale change loss on v vs "
                        "u_t (each scale = one more x2 avg-pool = lower frequency). "
                        "e.g. '1 1 1' scores full-res + 2 coarser scales, putting "
                        "gradient on the coherent edit and away from the high-freq "
                        "registration speckle a single-scale MSE chases. Unset = "
                        "plain full-resolution MSE. Bridge only.")
    p.add_argument("--edge_downweight", type=float, default=0.0,
                   help="gamma for softly down-weighting the full-res regression on "
                        "x_pre's structural edges (where misregistration fakes "
                        "change; the change-ROI is 2.2x enriched there). 0 = off; "
                        "try 0.3. Applied only at the full-resolution scale.")
    p.add_argument("--gate_init_bias", type=float, default=-2.0,
                   help="Initial bias of the gate logit (sigmoid(-2)~0.12), so the "
                        "gate starts sparse — a sensible prior over a near-identity "
                        "volume. Only used with --gated.")
    p.add_argument("--change_roi_frac", type=float, default=0.05,
                   help="Fraction of most-changed brain voxels that define the "
                        "change-region ROI reported at validation (change/* metrics). "
                        "This is where a model can actually beat identity; the "
                        "whole-volume metrics are blind to it.")
    # ── adversarial detail term (the anti-blur fix) ──────────────────────────
    p.add_argument("--adv_weight", type=float, default=0.0,
                   help="Weight of the adversarial (hinge PatchGAN) term on the "
                        "one-step prediction. 0 = off (pure regression). The CFM "
                        "regression stays the dominant loss and anchors structure; "
                        "the discriminator only adds high-frequency detail on top, "
                        "so keep this small (try 0.05). Bridge only (--source pre).")
    p.add_argument("--adv_lr", type=float, default=2e-4,
                   help="Discriminator learning rate (its own AdamW).")
    p.add_argument("--adv_warmup_epochs", type=int, default=25,
                   help="Train the generator on CFM alone for this many epochs "
                        "before switching the adversarial term on, so it edits a "
                        "coherent structure rather than fighting a discriminator "
                        "from noise. Only matters when --adv_weight > 0.")
    p.add_argument("--disc_dim",    type=int, default=32,
                   help="Base channel width of the PatchGAN discriminator.")
    p.add_argument("--disc_layers", type=int, default=3,
                   help="Number of stride-2 blocks in the discriminator (receptive "
                        "field / patch size). More = larger patches, coarser.")
    p.add_argument("--l1_weight",   type=float, default=0.0,
                   help="Weight of an L1 term on the velocity residual (added to MSE)")
    p.add_argument("--ssim_weight", type=float, default=0.0,
                   help="Weight of (1-SSIM3D) on the one-step x1 estimate")
    p.add_argument("--cfg_drop_prob",  type=float, default=0.0,
                   help="Conditioning dropout. Only coherent for --source noise.")
    p.add_argument("--guidance_scale", type=float, default=1.0,
                   help="CFG scale at sampling. Leave at 1.0 for --source pre.")
    # ── sampling / solver ────────────────────────────────────────────────────
    p.add_argument("--steps",  type=int, default=8,
                   help="ODE integration steps at sampling time")
    p.add_argument("--solver", type=str, default="heun", choices=["euler", "heun", "rk4"])
    p.add_argument("--val_steps", type=int, default=None,
                   help="Integration steps for validation metrics (default: --steps)")
    p.add_argument("--init_noise", type=float, default=0.0,
                   help="Std of noise added to x(0) at sampling; >0 gives an ensemble")
    # ── velocity network ─────────────────────────────────────────────────────
    p.add_argument("--dim",              type=int, default=32)
    p.add_argument("--dim_mults",        type=int, nargs="+", default=[1, 2, 4, 8])
    p.add_argument("--init_kernel_size", type=int, default=7,
                   help="Reduce to 3 on GPUs with <12 GB")
    p.add_argument("--resnet_groups",    type=int, default=8)
    p.add_argument("--zero_init_out",    dest="zero_init_out", action="store_true",
                   help="Zero the output conv so the untrained model returns x_pre "
                        "exactly, i.e. training starts at the identity baseline")
    p.add_argument("--no-zero-init-out", dest="zero_init_out", action="store_false")
    p.set_defaults(zero_init_out=True)
    # ── optimisation ─────────────────────────────────────────────────────────
    p.add_argument("--epochs",      type=int,   default=500)
    p.add_argument("--batch_size",  type=int,   default=2)
    p.add_argument("--lr",          type=float, default=2e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--warmup_epochs", type=int, default=10)
    p.add_argument("--min_lr_frac",   type=float, default=0.05,
                   help="Cosine schedule floor, as a fraction of --lr")
    p.add_argument("--grad_clip",   type=float, default=1.0)
    p.add_argument("--ema_beta",    type=float, default=0.999)
    p.add_argument("--ema_warmup",  type=int,   default=500,
                   help="Updates over which the EMA decay ramps up from 0")
    p.add_argument("--amp",    dest="amp", action="store_true")
    p.add_argument("--no-amp", dest="amp", action="store_false")
    p.set_defaults(amp=True)
    p.add_argument("--tf32",    dest="tf32", action="store_true")
    p.add_argument("--no-tf32", dest="tf32", action="store_false")
    p.set_defaults(tf32=True)
    p.add_argument("--persistent_workers",    dest="persistent_workers", action="store_true")
    p.add_argument("--no-persistent-workers", dest="persistent_workers", action="store_false")
    p.set_defaults(persistent_workers=True)
    # ── cadence ──────────────────────────────────────────────────────────────
    p.add_argument("--metric_every", type=int, default=5,
                   help="Run full sampling + MAE/MSE/PSNR/SSIM every N epochs. "
                        "val_loss is logged every epoch regardless.")
    p.add_argument("--vis_every",    type=int, default=5,
                   help="Save a val prediction grid every N epochs (matches the "
                        "7TCDM-3D cadence). Sampling here is cheap (~16 NFE).")
    # ── W&B ──────────────────────────────────────────────────────────────────
    p.add_argument("--wandb",    dest="wandb", action="store_true")
    p.add_argument("--no-wandb", dest="wandb", action="store_false")
    p.set_defaults(wandb=True)
    p.add_argument("--wandb_project", type=str, default="moyamoya-7tcdm3d",
                   help="Shares the existing 7TCDM-3D project for now; flow3d runs "
                        "are distinguished by --wandb_group (e.g. flow3d_bridge).")
    p.add_argument("--wandb_entity",  type=str, default=None)
    p.add_argument("--wandb_group",   type=str, default=None)
    p.add_argument("--wandb_mode",    type=str, default="online",
                   choices=["online", "offline", "disabled"])
    return p.parse_args()


def lr_at(epoch: int, args) -> float:
    """Linear warmup then cosine decay to ``min_lr_frac * lr``."""
    if epoch <= args.warmup_epochs and args.warmup_epochs > 0:
        return args.lr * epoch / max(args.warmup_epochs, 1)
    prog = (epoch - args.warmup_epochs) / max(args.epochs - args.warmup_epochs, 1)
    prog = min(max(prog, 0.0), 1.0)
    cos = 0.5 * (1 + np.cos(np.pi * prog))
    return args.lr * (args.min_lr_frac + (1 - args.min_lr_frac) * cos)


def report_gpu(device) -> None:
    """Print free VRAM from torch, not nvidia-smi.

    nvidia-smi is broken on this host (NVML version mismatch), and when the
    chosen GPU is already full the OOM surfaces as a cryptic
    ``NVML_SUCCESS ... INTERNAL ASSERT`` rather than an out-of-memory error —
    so the free figure is worth printing before anything allocates.
    """
    if device.type != "cuda":
        print(f"Device: {device}")
        return
    free, total = torch.cuda.mem_get_info(device.index or 0)
    print(f"Device: {device} ({torch.cuda.get_device_name(device)}) — "
          f"{free / 1e9:.1f} / {total / 1e9:.1f} GB free")
    if free < 12e9:
        print(f"  ! Only {free / 1e9:.1f} GB free on {device}. Full-resolution "
              f"training needs ~14 GB at --batch_size 2. Set GPU=<n> to pick "
              f"another device, or lower --dim / --init_kernel_size.")


CHANGE_KEYS = ("change_mae", "change_psnr", "change_ssim",
               "identity_change_mae", "change_mae_improvement", "change_roi_frac")


def save_gate_vis(x_pre_np, gate_np, target_np, title, path):
    """3 orthogonal views × [pre-op | predicted gate | true coherent-change].

    Lets you *see* whether the detection head fires where the scan actually
    changes — the direct read on the "learn WHERE" objective, which the scalar
    gate AUC only summarises. Gate/target are shown on a fixed 0…1 scale.
    """
    import matplotlib.pyplot as plt
    a = np.asarray(x_pre_np).squeeze(); g = np.asarray(gate_np).squeeze()
    t = np.asarray(target_np).squeeze()
    D, H, W = a.shape
    views = [("Axial", a[D // 2], g[D // 2], t[D // 2]),
             ("Coronal", a[:, H // 2], g[:, H // 2], t[:, H // 2]),
             ("Sagittal", a[:, :, W // 2], g[:, :, W // 2], t[:, :, W // 2])]
    cols = ["Pre-surgery", "Predicted gate (WHERE)", "True coherent change"]
    fig, axes = plt.subplots(3, 3, figsize=(10, 10), constrained_layout=True)
    for r, (name, pa, pg, pt) in enumerate(views):
        for c, (img, cmap, vmax) in enumerate([(pa, "gray", None),
                                               (pg, "inferno", 1.0),
                                               (pt, "inferno", 1.0)]):
            ax = axes[r, c]
            ax.imshow(img, cmap=cmap, origin="lower",
                      vmin=0 if vmax else None, vmax=vmax)
            ax.axis("off")
            if r == 0:
                ax.set_title(cols[c], fontsize=11)
        axes[r, 0].set_ylabel(name, fontsize=10)
    fig.suptitle(title, fontsize=12)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


@torch.no_grad()
def evaluate(model, val_dl, device, args, steps: int):
    """Sampled predictions over the val split → (whole-volume, change-region) means.

    Whole-volume metrics are the union-mask numbers the 7TCDM3D/LDM models report
    (comparable, but ~71% background). Change-region metrics score only the ROI
    where surgery actually altered the scan (see metrics.change_region_report) —
    the number that reveals whether the model learned the substantial edits.
    """
    acc = {"mae": [], "mse": [], "psnr": [], "ssim": []}
    cacc = {k: [] for k in CHANGE_KEYS}
    gated = getattr(model, "gated", False)
    gate_corr, gate_auc = [], []
    for x, y in val_dl:
        x_d = x.to(device, non_blocking=True)
        pred = model.sample(x_d, steps=steps, solver=args.solver,
                            guidance_scale=args.guidance_scale,
                            init_noise=args.init_noise)
        gmap = model.predict_change_map(x_d) if gated else None
        for i in range(x.shape[0]):
            p = pred[i].float().cpu()
            # Union (whole-volume) mask, exactly as train_ldm_7tcdm3d scores —
            # `(x != 0) | (y != 0)`. With zero_background off this is every voxel.
            m = compute_metrics(p, y[i], union_mask(x[i], y[i]))
            for k in acc:
                acc[k].append(m[k])
            cm = change_region_report(p, x[i], y[i], frac=args.change_roi_frac)
            for k in cacc:
                cacc[k].append(cm[k])
            if gmap is not None:
                gc, ga = _gate_quality(gmap[i], x[i], y[i], args.det_sigma,
                                       args.change_roi_frac)
                if np.isfinite(gc):
                    gate_corr.append(gc)
                if np.isfinite(ga):
                    gate_auc.append(ga)
    whole  = {k: float(np.mean(v)) for k, v in acc.items()}
    # nanmean: a subject with an empty ROI (no change) contributes NaN, skip it.
    change = {k: (float(np.nanmean(v)) if len(v) else float("nan"))
              for k, v in cacc.items()}
    change["gate_corr"] = float(np.mean(gate_corr)) if gate_corr else float("nan")
    change["gate_auc"]  = float(np.mean(gate_auc))  if gate_auc  else float("nan")
    return whole, change


def _gate_quality(gate, x_pre, x_post, det_sigma, roi_frac):
    """How well the predicted gate localises the *true* coherent change (brain only).

    Returns ``(pearson_corr, roc_auc)``:
      * corr — Pearson r between the gate and the smoothed-|Δ| soft target,
      * auc  — ROC-AUC of the gate as a detector of the top-``roi_frac`` coherent-
        change voxels (0.5 = chance, 1.0 = perfect ranking of WHERE it changes).
    Both restricted to brain voxels so the near-constant background can't inflate
    them. This is the headline "did it learn WHERE" number.
    """
    g = gate.float().cpu().numpy().squeeze()
    tgt = coherent_change_target(x_pre.unsqueeze(0), x_post.unsqueeze(0),
                                 det_sigma)[0].numpy().squeeze()
    fg = foreground_mask(x_pre, x_post)
    gf, tf = g[fg], tgt[fg]
    corr = float(np.corrcoef(gf, tf)[0, 1]) if gf.std() > 0 and tf.std() > 0 else float("nan")
    # binary labels = top-frac coherent-change voxels; rank-based AUC via Mann-Whitney
    thr = np.quantile(tf, 1.0 - roi_frac)
    pos = gf[tf >= thr]; neg = gf[tf < thr]
    if len(pos) == 0 or len(neg) == 0:
        return corr, float("nan")
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64); ranks[order] = np.arange(1, len(order) + 1)
    auc = (ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))
    return corr, float(auc)


def main():
    args = get_args()
    args.seed = resolve_seed(args.seed)
    seed_everything(args.seed)

    out_dir = Path(args.out_dir or f"runs/flow3d_{datetime.now():%Y-%m-%d_%H-%M-%S}")
    out_dir.mkdir(parents=True, exist_ok=True)
    args.out_dir = str(out_dir)
    install_run_logger(out_dir, "train")
    print(f"[Flow3D] logging to {out_dir}")
    print(f"Seed: {args.seed}  (rerun with --seed {args.seed} to reproduce)")

    if args.tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    device = get_device()
    report_gpu(device)

    if args.source == "pre" and args.guidance_scale != 1.0:
        print(f"  ! --guidance_scale {args.guidance_scale} with --source pre: the "
              f"conditioning also seeds the trajectory, so CFG is not well defined "
              f"here. Use --source noise if you want guidance.")

    if args.source == "pre" and args.augment and args.aug_noise_std > 0:
        # For the diffusion models x_pre is only a conditioning signal, so noising
        # it is a plain robustness augmentation. For the bridge it is also x(0),
        # so training would learn to transport a *noisy* start to a clean target
        # while sampling always starts from a clean one — the model picks up a
        # denoising component it then applies to an image that was never noisy,
        # over-smoothing exactly the residual we are trying to resolve.
        print(f"  ! --aug_noise_std {args.aug_noise_std} with --source pre noises the "
              f"ODE's starting point, not just the conditioning, which biases "
              f"sampling (it starts from a clean x_pre). Use 0 unless testing this.")

    # ── data ─────────────────────────────────────────────────────────────────
    if args.augment:
        print(f"[augment] flip_p={args.aug_flip_p} rotate_deg={args.aug_rotate_deg} "
              f"noise_std={args.aug_noise_std} (input-only)")
    else:
        print("[augment] disabled (--no-augment)")
    train_dl, val_dl = build_loaders(args, augment=args.augment)
    n_train, n_val = len(train_dl.dataset), len(val_dl.dataset)
    print(f"  Train {n_train} / Val {n_val} samples")

    # ── the number to beat ───────────────────────────────────────────────────
    # Computed on this run's own validation split, before anything is trained,
    # with the same whole-volume union mask the metrics use so the two are
    # directly comparable.
    baseline = identity_baseline(val_dl, mask_fn=union_mask)
    print("\n[identity baseline] predict x_post := x_pre, on this val split:")
    print("   " + "  ".join(f"{k.upper()}={v:.4f}" for k, v in baseline.items()))
    print("   A model that does not clear this is worse than doing nothing.")
    if args.zero_background:
        print("   (--zero_background is ON: exact-zero background + whole-volume "
              "union mask. Note this is NOT the 7TCDM3D/LDM convention.)\n")
    else:
        print("   (whole-volume, matching the 7TCDM3D / LDM models: the whole "
              "z-scored volume is scored, background included — the same footing "
              "those models are reported on. ~71% of every metric is background.)\n")

    save_hparams(args, "flow3d", out_dir, extra={
        "dataset": {"data_root": args.data_root, "n_train": n_train, "n_val": n_val,
                    "val_frac": args.val_frac, "seed": args.seed,
                    "n_folds": args.n_folds, "fold": args.fold},
        "identity_baseline": baseline,
    })
    run = init_wandb(args, out_dir, args.wandb_project,
                     config_extra={"n_train": n_train, "n_val": n_val,
                                   **{f"baseline_{k}": v for k, v in baseline.items()}})

    # ── model ────────────────────────────────────────────────────────────────
    model = build_flow3d(
        dim=args.dim, dim_mults=tuple(args.dim_mults),
        init_kernel_size=args.init_kernel_size, resnet_groups=args.resnet_groups,
        zero_init_out=args.zero_init_out,
        source=args.source, condition=args.condition, sigma=args.sigma,
        t_dist=args.t_dist,
        logit_mean=args.logit_mean, logit_std=args.logit_std,
        change_weight=args.change_weight,
        l1_weight=args.l1_weight, ssim_weight=args.ssim_weight,
        cfg_drop_prob=args.cfg_drop_prob,
        gated=args.gated, det_weight=args.det_weight, det_sigma=args.det_sigma,
        ms_weights=tuple(args.ms_weights) if args.ms_weights else None,
        edge_downweight=args.edge_downweight, gate_init_bias=args.gate_init_bias,
        steps=args.steps, solver=args.solver,
    ).to(device)
    # Record what --condition actually resolved to, so the checkpoint rebuilds
    # the right input width and hparams.json is not ambiguous.
    args.condition = model.condition

    if args.source == "pre" and args.condition == "concat" and args.sigma == 0.0:
        print("  ! --source pre --condition concat with --sigma 0 lets the network "
              "solve the path algebraically as (x_t-x_pre)/t: training loss will "
              "look excellent and sampling will collapse to the input. Use --sigma>0.")
    if args.condition == "none":
        print("  ! --condition none: the velocity net never sees x_pre, so it can "
              "only learn E[u_t | x_t] and the loss barely descends. This is an "
              "ablation; use --condition concat for real training.")
    if args.source == "noise" and args.change_weight > 0:
        print(f"  ! --change_weight {args.change_weight} is ignored for --source noise "
              f"(there the target velocity is x_post−noise, not the change map). The "
              f"change weighting only applies to the bridge (--source pre).")

    # ── change-detection guardrails ──────────────────────────────────────────
    if args.gated and args.det_weight <= 0:
        print("  ! --gated without --det_weight: the gate is architectural but "
              "unsupervised, so it just absorbs into the residual. Set --det_weight "
              "(try 0.3) to actually train the WHERE detector. Continuing as an ablation.")
    if args.det_weight > 0 and not args.gated:
        print(f"  ! --det_weight {args.det_weight} needs --gated (the detection loss "
              f"scores the gate map, which only exists on a gated net). Ignored.")
    if args.det_weight > 0 and args.source != "pre":
        print(f"  ! --det_weight {args.det_weight} needs --source pre (the coherent-"
              f"change target is x_post−x_pre). Ignored.")
    if (args.gated or args.ms_weights) and args.change_weight > 0:
        print(f"  ! coherent-change training with --change_weight {args.change_weight} "
              f">0: the raw-|Δ| voxel weight up-weights the noisy edge tail, which is "
              f"exactly what the multi-scale/detection terms exist to avoid. Consider "
              f"--change_weight 0.")
    if args.gated:
        print(f"[gated] velocity = gate·residual | det_weight={args.det_weight} "
              f"det_sigma={args.det_sigma} ms_weights={args.ms_weights} "
              f"edge_downweight={args.edge_downweight}")

    # EMA copy, sampled from at validation time.
    ema_model = deepcopy(model).to(device).eval()
    for p in ema_model.parameters():
        p.requires_grad_(False)
    ema = EMA(beta=args.ema_beta, warmup=args.ema_warmup)

    n_params = sum(p.numel() for p in model.net.parameters())
    print(f"[Flow3D] source={args.source} condition={args.condition} "
          f"sigma={args.sigma} solver={args.solver} steps={args.steps} | "
          f"velocity net params: {n_params:,}")

    opt = torch.optim.AdamW(model.net.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)

    # ── adversarial detail term (optional) ─────────────────────────────────────
    # A conditional PatchGAN discriminator judges the one-step prediction
    # x_pre+v(x_pre,0) against the real x_post. It fights the mode-averaging blur
    # that a pure regression loss produces — the reason a change-weighted MSE still
    # only made timid, smeared edits. The CFM regression stays dominant and keeps
    # the output faithful; the discriminator only adds high-frequency detail.
    disc = opt_d = None
    if args.adv_weight > 0.0:
        if args.source != "pre":
            raise SystemExit("--adv_weight > 0 requires --source pre (the adversarial "
                             "term scores the bridge's one-step prediction).")
        disc = build_discriminator3d(dim=args.disc_dim, n_layers=args.disc_layers).to(device)
        opt_d = torch.optim.AdamW(disc.parameters(), lr=args.adv_lr,
                                  betas=(0.5, 0.9), weight_decay=0.0)
        n_d = sum(p.numel() for p in disc.parameters())
        print(f"[adversarial] hinge PatchGAN ON: weight={args.adv_weight} "
              f"lr={args.adv_lr} warmup={args.adv_warmup_epochs}ep | "
              f"disc params: {n_d:,} (dim={args.disc_dim}, layers={args.disc_layers})")
        print("  ! adds a 2nd generator forward + a discriminator per step; if it "
              "OOMs, lower --batch_size or --dim (A6000 is fine at bs2).")

    val_steps = args.val_steps or args.steps
    nfe = {"euler": 1, "heun": 2, "rk4": 4}[args.solver] * val_steps
    print(f"  Metrics every {args.metric_every} epoch(s): {val_steps} {args.solver} "
          f"steps = {nfe} NFE per sample (DDIM baseline was 50)")

    csv_log = MetricsCSV(out_dir / "metrics.csv",
                         ["epoch", "lr", "train_loss", "reg_loss", "det_loss",
                          "g_adv", "d_loss", "val_loss",
                          "mae", "mse", "psnr", "ssim",
                          "change_mae", "identity_change_mae",
                          "change_mae_improvement", "change_psnr", "change_ssim",
                          "gate_corr", "gate_auc"])
    vis_dir = out_dir / "vis"
    vis_dir.mkdir(exist_ok=True)

    best_val = float("inf")
    best_change = float("-inf")   # best change-ROI improvement over identity
    best_gate = float("-inf")     # best change-localisation AUC (gated model)
    best_metric = {k: (float("-inf") if hib else float("inf"))
                   for k, hib in METRIC_HIGHER_BETTER.items()}

    def checkpoint():
        return {"model": model.net.state_dict(),
                "ema":   ema_model.net.state_dict(),
                "disc":  disc.state_dict() if disc is not None else None,
                "args":  vars(args),
                "identity_baseline": baseline}

    # ── training loop ────────────────────────────────────────────────────────
    for epoch in range(1, args.epochs + 1):
        lr = lr_at(epoch, args)
        for g in opt.param_groups:
            g["lr"] = lr

        model.train()
        if disc is not None:
            disc.train()
        tr_loss = tr_adv = tr_d = tr_reg = tr_det = 0.0
        # Let the generator learn a coherent coarse mapping before the
        # discriminator starts pushing on it (see --adv_warmup_epochs).
        adv_on = disc is not None and epoch > args.adv_warmup_epochs

        def _ac():
            return torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                  enabled=args.amp)

        for x, y in train_dl:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            # ── generator: CFM regression (always the dominant term) ──────────
            opt.zero_grad(set_to_none=True)
            with _ac():
                loss_cfm, comp = model.flow_loss(x_post=y, x_pre=x,
                                                 return_components=True)
            loss_cfm.backward()                       # frees the CFM graph
            tr_loss += loss_cfm.item()
            tr_reg += comp.get("reg", float("nan"))
            tr_det += comp.get("det", 0.0)

            # ── adversarial detail term (after warmup) ───────────────────────
            if adv_on:
                with _ac():
                    x1 = model.onestep_prediction(x)          # differentiable fake
                # discriminator step: real x_post vs the detached prediction
                with _ac():
                    loss_d = d_hinge_loss(disc(x, y).float(),
                                          disc(x, x1.detach()).float())
                opt_d.zero_grad(set_to_none=True)
                loss_d.backward()
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(disc.parameters(), args.grad_clip)
                opt_d.step()
                tr_d += loss_d.item()
                # generator adversarial gradient — reuse x1's graph (D just stepped),
                # accumulated into the same opt as the CFM grads above.
                with _ac():
                    g_adv = g_hinge_loss(disc(x, x1).float())
                (args.adv_weight * g_adv).backward()
                tr_adv += g_adv.item()

            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.net.parameters(), args.grad_clip)
            opt.step()
            ema.update(ema_model.net, model.net)
        nb = max(len(train_dl), 1)
        tr_loss /= nb; tr_adv /= nb; tr_d /= nb; tr_reg /= nb; tr_det /= nb
        adv_str = f"  adv={tr_adv:+.3f} d={tr_d:.3f}" if adv_on else ""
        det_str = f"  det={tr_det:.3f}" if (args.gated and args.det_weight > 0) else ""

        # ── validation ───────────────────────────────────────────────────────
        do_metrics = (args.metric_every <= 1
                      or epoch % args.metric_every == 0
                      or epoch == args.epochs)
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_dl:
                x_d = x.to(device, non_blocking=True)
                y_d = y.to(device, non_blocking=True)
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                    enabled=args.amp):
                    val_loss += model.flow_loss(x_post=y_d, x_pre=x_d).item()
        val_loss /= max(len(val_dl), 1)

        # Sampled metrics use the EMA weights — that is what eval/inference load.
        m, cm = (evaluate(ema_model, val_dl, device, args, val_steps) if do_metrics
                 else ({k: float("nan") for k in METRIC_HIGHER_BETTER},
                       {k: float("nan") for k in (*CHANGE_KEYS, "gate_corr", "gate_auc")}))

        if do_metrics:
            flags = "".join(
                ("+" if ((m[k] > baseline[k]) if hib else (m[k] < baseline[k])) else "-")
                for k, hib in METRIC_HIGHER_BETTER.items())
            print(f"Epoch {epoch:4d}/{args.epochs}  lr={lr:.2e}  "
                  f"train={tr_loss:.4f}{det_str}{adv_str}  val={val_loss:.4f}  "
                  f"MAE={m['mae']:.4f}  MSE={m['mse']:.4f}  "
                  f"PSNR={m['psnr']:.2f}  SSIM={m['ssim']:.4f}  "
                  f"[vs identity: {flags}]")
            # The number the whole-volume metrics can't show: error in the region
            # that actually changed, and whether it beats copying x_pre there.
            gate_str = (f"  |  gate AUC={cm['gate_auc']:.3f} corr={cm['gate_corr']:.3f}"
                        if args.gated and np.isfinite(cm.get("gate_auc", float("nan")))
                        else "")
            print(f"            change-ROI (top {args.change_roi_frac:.0%}): "
                  f"MAE={cm['change_mae']:.4f} vs identity {cm['identity_change_mae']:.4f} "
                  f"(Δ={cm['change_mae_improvement']:+.4f}, "
                  f"{'BEATS' if cm['change_mae_improvement'] > 0 else 'loses to'} identity)  "
                  f"PSNR={cm['change_psnr']:.2f}  SSIM={cm['change_ssim']:.4f}{gate_str}")
        else:
            print(f"Epoch {epoch:4d}/{args.epochs}  lr={lr:.2e}  "
                  f"train={tr_loss:.4f}{det_str}{adv_str}  val={val_loss:.4f}")

        csv_log.append({"epoch": epoch, "lr": lr, "train_loss": tr_loss,
                        "reg_loss": tr_reg, "det_loss": tr_det,
                        "g_adv": tr_adv, "d_loss": tr_d,
                        "val_loss": val_loss, **m, **cm})
        log = {"train_loss": tr_loss, "val_loss": val_loss, "lr": lr}
        if args.gated and args.det_weight > 0:
            log["train/reg"] = tr_reg
            log["train/det"] = tr_det
        if disc is not None:
            log["adv/g_adv"] = tr_adv
            log["adv/d_loss"] = tr_d
            log["adv/on"] = float(adv_on)
        if do_metrics:
            log.update(m)
            # Overlay the identity baseline as a constant and log the signed
            # improvement (positive = better than copying x_pre) so the W&B
            # charts show progress against the number that actually matters,
            # not just an absolute metric that could still be losing.
            for k, hib in METRIC_HIGHER_BETTER.items():
                log[f"identity/{k}"] = baseline[k]
                log[f"improvement/{k}"] = (m[k] - baseline[k]) if hib else (baseline[k] - m[k])
            log["beats_identity_count"] = sum(
                1 for k, hib in METRIC_HIGHER_BETTER.items()
                if ((m[k] > baseline[k]) if hib else (m[k] < baseline[k])))
            # Change-region panel: the model's error where surgery edits the scan,
            # the identity error there, and the signed improvement (the headline).
            log["change/mae"]           = cm["change_mae"]
            log["change/psnr"]          = cm["change_psnr"]
            log["change/ssim"]          = cm["change_ssim"]
            log["change/identity_mae"]  = cm["identity_change_mae"]
            log["change/mae_improvement"] = cm["change_mae_improvement"]
            log["change/roi_frac"]      = cm["change_roi_frac"]
            # Detection head: did it learn WHERE the change is? (gated model only)
            if args.gated and np.isfinite(cm.get("gate_auc", float("nan"))):
                log["gate/auc"]  = cm["gate_auc"]
                log["gate/corr"] = cm["gate_corr"]
        wandb_log(run, log, step=epoch)

        ckpt = checkpoint()
        torch.save(ckpt, out_dir / "last.pt")
        if val_loss < best_val:
            best_val = val_loss
            torch.save(ckpt, out_dir / "best.pt")
        if do_metrics:
            for k, hib in METRIC_HIGHER_BETTER.items():
                if (m[k] > best_metric[k]) if hib else (m[k] < best_metric[k]):
                    best_metric[k] = m[k]
                    torch.save(ckpt, out_dir / f"best_{k}.pt")
                    print(f"  → new best {k} ({m[k]:.4f})")
            # Select on the metric that actually reflects learning the edits.
            ci = cm["change_mae_improvement"]
            if np.isfinite(ci) and ci > best_change:
                best_change = ci
                torch.save(ckpt, out_dir / "best_change.pt")
                print(f"  → new best change-ROI improvement ({ci:+.4f})")
            # And on how well the gate localises WHERE the change is (gated model).
            ga = cm.get("gate_auc", float("nan"))
            if np.isfinite(ga) and ga > best_gate:
                best_gate = ga
                torch.save(ckpt, out_dir / "best_gate.pt")
                print(f"  → new best gate AUC ({ga:.4f})")

        # ── visualisation ────────────────────────────────────────────────────
        if args.vis_every > 0 and epoch % args.vis_every == 0:
            ds = val_dl.dataset
            x_v, y_v = ds[random.randint(0, len(ds) - 1)]
            xb = x_v.unsqueeze(0).to(device)
            with torch.no_grad():
                pb = ema_model.sample(xb, steps=val_steps, solver=args.solver,
                                      guidance_scale=args.guidance_scale,
                                      init_noise=args.init_noise)
            path = vis_dir / f"epoch_{epoch:04d}.png"
            save_grid(_to_np(xb), _to_np(y_v.unsqueeze(0)), _to_np(pb),
                      f"Flow3D ({args.source}) — epoch {epoch}", path)
            print(f"  → saved visualisation: {path}")
            wandb_log_image(run, "val_sample", path, step=epoch)

            # Gate panel: predicted change-localisation vs the true coherent change.
            if args.gated:
                with torch.no_grad():
                    gate = ema_model.predict_change_map(xb)
                tgt = coherent_change_target(x_v.unsqueeze(0), y_v.unsqueeze(0),
                                             args.det_sigma)
                gpath = vis_dir / f"epoch_{epoch:04d}_gate.png"
                save_gate_vis(_to_np(x_v.unsqueeze(0)), gate[0].cpu().numpy(),
                              tgt[0].cpu().numpy(),
                              f"Flow3D gate (WHERE) — epoch {epoch}", gpath)
                print(f"  → saved gate visualisation: {gpath}")
                wandb_log_image(run, "val_gate", gpath, step=epoch)

    # ── wrap up ──────────────────────────────────────────────────────────────
    print("\n=== final vs identity baseline ===")
    print(f"{'metric':>7}  {'best model':>11}  {'identity':>9}  verdict")
    for k, hib in METRIC_HIGHER_BETTER.items():
        beat = (best_metric[k] > baseline[k]) if hib else (best_metric[k] < baseline[k])
        print(f"{k.upper():>7}  {best_metric[k]:11.4f}  {baseline[k]:9.4f}  "
              f"{'BEATS identity' if beat else 'loses to identity'}")
    # The headline for this branch: did the model reduce error where surgery
    # actually changed the scan? (Whole-volume metrics above can't show it.)
    print(f"\nBest change-ROI MAE improvement over identity (copy x_pre): "
          f"{best_change:+.4f}  → checkpoint best_change.pt")
    print("  (positive ⇒ the model genuinely edited the region of change, which is "
          "the goal here; ~0 ⇒ it collapsed to copying x_pre.)")
    if args.gated:
        print(f"Best change-localisation gate AUC: {best_gate:.4f}  → best_gate.pt")
        print("  (0.5 ⇒ the gate is chance at finding WHERE the scan changes; "
              "→1.0 ⇒ it localises the surgical edit. The 'learn where' headline.)")
    plot_progression(csv_log, out_dir, f"{out_dir.name} — Flow3D ({args.source})",
                     baseline=baseline)
    print(f"\nDone. Checkpoints in {out_dir}")
    wandb_finish(run)


if __name__ == "__main__":
    main()
