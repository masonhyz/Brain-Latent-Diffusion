"""
Flow3D training — conditional flow matching for paired pre→post CBF prediction.

Single stage, image space, no autoencoder. See moyamoya/models/flow3d.py for
why the default transports x_pre → x_post rather than noise → x_post.

    # default: bridge from the pre-op volume (recommended)
    python scripts/train_flow3d.py --out_dir runs/flow3d

    # ablation: standard CFM from Gaussian noise, same U-Net, for a like-for-like
    # comparison against the diffusion models
    python scripts/train_flow3d.py --source noise --sigma 0 --cfg_drop_prob 0.15 \
        --guidance_scale 3.0 --out_dir runs/flow3d_noise

Every run prints, and plots, the identity baseline (predict x_post := x_pre) on
its own validation split. On this dataset that baseline beats every diffusion
model in the repo, so it is the number that matters.
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
    change_region_report, compute_metrics, identity_baseline, union_mask,
)
from moyamoya.models.flow3d import (
    EMA, build_discriminator3d, build_flow3d, d_hinge_loss, g_hinge_loss,
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
                        "MSE (ablation). Only applied for --source pre.")
    p.add_argument("--change_sampler_beta", type=float, default=0.0,
                   help="Oversample big-change subjects with a WeightedRandomSampler, "
                        "weight ∝ 1+beta*g/median(g) with g the subject's global "
                        "change. 0 = uniform. Keep modest (<=1) with ~200 subjects — "
                        "the per-voxel --change_weight is the safer primary lever; "
                        "this is the complementary coarse one (your 'focus on the "
                        "pairs that differ' idea).")
    p.add_argument("--change_roi_frac", type=float, default=0.05,
                   help="Fraction of most-changed brain voxels that define the "
                        "change-region ROI reported at validation (change/* metrics). "
                        "This is where a model can actually beat identity; the "
                        "whole-volume metrics are blind to it.")
    p.add_argument("--coherent", dest="coherent", action="store_true",
                   help="Additionally report the coherent (registration-noise-"
                        "stripped) change metrics coherent_{mae,mse,psnr,ssim} + "
                        "coherent_frac + edge_enrichment. Off by default; the raw "
                        "change/* metrics are reported either way.")
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


# Raw change-ROI keys, always reported. The coherent (registration-noise-stripped)
# family is opt-in via --coherent; see metrics.change_region_report / README §3.
CHANGE_KEYS = ("change_mae", "change_psnr", "change_ssim",
               "identity_change_mae", "change_mae_improvement", "change_roi_frac")
COHERENT_KEYS = ("coherent_mae", "coherent_mse", "coherent_psnr", "coherent_ssim",
                 "identity_coherent_mae", "coherent_mae_improvement",
                 "coherent_frac", "edge_enrichment")


def change_keys_for(args) -> tuple:
    """The change-region keys reported for this run: raw ROI always, coherent
    family only when ``--coherent`` is set."""
    return CHANGE_KEYS + (COHERENT_KEYS if getattr(args, "coherent", False) else ())


@torch.no_grad()
def evaluate(model, val_dl, device, args, steps: int):
    """Sampled predictions over the val split → (whole-volume, change-region) means.

    Whole-volume metrics are the union-mask numbers the 7TCDM3D/LDM models report
    (comparable, but ~71% background). Change-region metrics score only the ROI
    where surgery actually altered the scan (see metrics.change_region_report) —
    the number that reveals whether the model learned the substantial edits. The
    coherent (noise-stripped) family is added only under ``--coherent``.
    """
    acc = {"mae": [], "mse": [], "psnr": [], "ssim": []}
    cacc = {k: [] for k in change_keys_for(args)}
    for x, y in val_dl:
        x_d = x.to(device, non_blocking=True)
        pred = model.sample(x_d, steps=steps, solver=args.solver,
                            guidance_scale=args.guidance_scale,
                            init_noise=args.init_noise)
        for i in range(x.shape[0]):
            p = pred[i].float().cpu()
            # Union (whole-volume) mask, exactly as train_ldm_7tcdm3d scores —
            # `(x != 0) | (y != 0)`. With zero_background off this is every voxel.
            m = compute_metrics(p, y[i], union_mask(x[i], y[i]))
            for k in acc:
                acc[k].append(m[k])
            cm = change_region_report(p, x[i], y[i], frac=args.change_roi_frac,
                                      coherent=args.coherent)
            for k in cacc:
                cacc[k].append(cm[k])
    whole  = {k: float(np.mean(v)) for k, v in acc.items()}
    # nanmean: a subject with an empty ROI (no change) contributes NaN, skip it.
    change = {k: (float(np.nanmean(v)) if len(v) else float("nan"))
              for k, v in cacc.items()}
    return whole, change


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
        cfg_drop_prob=args.cfg_drop_prob, steps=args.steps, solver=args.solver,
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
                         ["epoch", "lr", "train_loss", "g_adv", "d_loss", "val_loss",
                          "mae", "mse", "psnr", "ssim",
                          *CHANGE_KEYS,
                          # coherent columns only when --coherent is set
                          *(COHERENT_KEYS if args.coherent else ())])
    vis_dir = out_dir / "vis"
    vis_dir.mkdir(exist_ok=True)

    best_val = float("inf")
    best_change = float("-inf")   # best change-ROI improvement over identity
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
        tr_loss = tr_adv = tr_d = 0.0
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
                loss_cfm = model.flow_loss(x_post=y, x_pre=x)
            loss_cfm.backward()                       # frees the CFM graph
            tr_loss += loss_cfm.item()

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
        tr_loss /= nb; tr_adv /= nb; tr_d /= nb
        adv_str = f"  adv={tr_adv:+.3f} d={tr_d:.3f}" if adv_on else ""

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
                       {k: float("nan") for k in change_keys_for(args)}))

        if do_metrics:
            flags = "".join(
                ("+" if ((m[k] > baseline[k]) if hib else (m[k] < baseline[k])) else "-")
                for k, hib in METRIC_HIGHER_BETTER.items())
            print(f"Epoch {epoch:4d}/{args.epochs}  lr={lr:.2e}  "
                  f"train={tr_loss:.4f}{adv_str}  val={val_loss:.4f}  "
                  f"MAE={m['mae']:.4f}  MSE={m['mse']:.4f}  "
                  f"PSNR={m['psnr']:.2f}  SSIM={m['ssim']:.4f}  "
                  f"[vs identity: {flags}]")
            # The number the whole-volume metrics can't show: error in the region
            # that actually changed, and whether it beats copying x_pre there.
            print(f"            change-ROI (top {args.change_roi_frac:.0%}): "
                  f"MAE={cm['change_mae']:.4f} vs identity {cm['identity_change_mae']:.4f} "
                  f"(Δ={cm['change_mae_improvement']:+.4f}, "
                  f"{'BEATS' if cm['change_mae_improvement'] > 0 else 'loses to'} identity)  "
                  f"PSNR={cm['change_psnr']:.2f}  SSIM={cm['change_ssim']:.4f}")
            if args.coherent:
                # Honest, noise-aware line: only ~coherent_frac of that ROI is
                # recoverable signal (the rest is registration/acq noise, edge-
                # enriched ~edge_enrichment×); coherent-Δ is skill on the part
                # that is actually predictable. Report THIS in the paper.
                print(f"              └ coherent {cm['coherent_frac']:.2f} of ROI "
                      f"(edge-enrich {cm['edge_enrichment']:.1f}×); coherent-MAE "
                      f"{cm['coherent_mae']:.4f} vs identity "
                      f"{cm['identity_coherent_mae']:.4f} "
                      f"(Δ={cm['coherent_mae_improvement']:+.4f})")
        else:
            print(f"Epoch {epoch:4d}/{args.epochs}  lr={lr:.2e}  "
                  f"train={tr_loss:.4f}{adv_str}  val={val_loss:.4f}")

        csv_log.append({"epoch": epoch, "lr": lr, "train_loss": tr_loss,
                        "g_adv": tr_adv, "d_loss": tr_d,
                        "val_loss": val_loss, **m, **cm})
        log = {"train_loss": tr_loss, "val_loss": val_loss, "lr": lr}
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
            if args.coherent:
                # Noise-aware panel: the recoverable fraction of the ROI, the
                # mis-registration signature, and skill on the coherent edit.
                log["change/coherent_frac"]        = cm["coherent_frac"]
                log["change/edge_enrichment"]      = cm["edge_enrichment"]
                log["change/coherent_mae"]         = cm["coherent_mae"]
                log["change/coherent_identity_mae"] = cm["identity_coherent_mae"]
                log["change/coherent_mae_improvement"] = cm["coherent_mae_improvement"]
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
    plot_progression(csv_log, out_dir, f"{out_dir.name} — Flow3D ({args.source})",
                     baseline=baseline)
    print(f"\nDone. Checkpoints in {out_dir}")
    wandb_finish(run)


if __name__ == "__main__":
    main()
