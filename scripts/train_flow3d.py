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
from moyamoya.metrics import compute_metrics, identity_baseline, tissue_mask
from moyamoya.models.flow3d import EMA, build_flow3d
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
                        "z-scoring (on by default here, off for the diffusion "
                        "models). Without it the background lands at ~-2.4 and "
                        "every (x != 0) brain mask selects the whole volume, so "
                        "~71%% of every metric measures a constant background.")
    p.add_argument("--no-zero-background", dest="zero_background", action="store_false")
    p.set_defaults(zero_background=True)
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
    p.add_argument("--condition", type=str, default=None, choices=["none", "concat"],
                   help="How x_pre reaches the network. Default resolves per "
                        "source: 'none' for the bridge (x_t already carries the "
                        "anatomy, and supplying x_pre lets the net solve the path "
                        "algebraically as (x_t-x_pre)/t instead of learning "
                        "anything), 'concat' for --source noise, where it is "
                        "indispensable. Override only to study the failure.")
    p.add_argument("--sigma",  type=float, default=0.1,
                   help="Bridge-noise scale in gamma(t)=sigma*sin(pi*t). Smooths "
                        "the velocity field around the path; 0 = deterministic path.")
    p.add_argument("--t_dist", type=str, default="uniform",
                   choices=["uniform", "logit_normal"],
                   help="Training-time distribution over t. logit_normal is the "
                        "SD3 schedule (concentrates on mid-path).")
    p.add_argument("--logit_mean", type=float, default=0.0)
    p.add_argument("--logit_std",  type=float, default=1.0)
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


@torch.no_grad()
def evaluate(model, val_dl, device, args, steps: int):
    """Sampled predictions over the val split → mean metrics."""
    acc = {"mae": [], "mse": [], "psnr": [], "ssim": []}
    for x, y in val_dl:
        x_d = x.to(device, non_blocking=True)
        pred = model.sample(x_d, steps=steps, solver=args.solver,
                            guidance_scale=args.guidance_scale,
                            init_noise=args.init_noise)
        for i in range(x.shape[0]):
            m = compute_metrics(pred[i].float().cpu(), y[i], tissue_mask(x[i], y[i]))
            for k in acc:
                acc[k].append(m[k])
    return {k: float(np.mean(v)) for k, v in acc.items()}


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
    # Computed on this run's own validation split, before anything is trained.
    baseline = identity_baseline(val_dl)
    print("\n[identity baseline] predict x_post := x_pre, on this val split:")
    print("   " + "  ".join(f"{k.upper()}={v:.4f}" for k, v in baseline.items()))
    print("   A model that does not clear this is worse than doing nothing.")
    if args.zero_background:
        print("   (brain tissue only — --zero_background makes the mask real. The "
              "diffusion models' historical numbers are whole-volume and score "
              "~2x better on MAE purely from the constant background.)\n")
    else:
        print("   (whole-volume: with --no-zero-background the mask selects every "
              "voxel, so ~71% of this is the constant background.)\n")

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
        l1_weight=args.l1_weight, ssim_weight=args.ssim_weight,
        cfg_drop_prob=args.cfg_drop_prob, steps=args.steps, solver=args.solver,
    ).to(device)
    # Record what --condition actually resolved to, so the checkpoint rebuilds
    # the right input width and hparams.json is not ambiguous.
    args.condition = model.condition

    if args.source == "pre" and args.condition == "concat":
        print("  ! --source pre with --condition concat lets the network solve "
              "the path algebraically as (x_t-x_pre)/t; training loss will look "
              "excellent and sampling will be no better than the input.")

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

    val_steps = args.val_steps or args.steps
    nfe = {"euler": 1, "heun": 2, "rk4": 4}[args.solver] * val_steps
    print(f"  Metrics every {args.metric_every} epoch(s): {val_steps} {args.solver} "
          f"steps = {nfe} NFE per sample (DDIM baseline was 50)")

    csv_log = MetricsCSV(out_dir / "metrics.csv",
                         ["epoch", "lr", "train_loss", "val_loss",
                          "mae", "mse", "psnr", "ssim"])
    vis_dir = out_dir / "vis"
    vis_dir.mkdir(exist_ok=True)

    best_val = float("inf")
    best_metric = {k: (float("-inf") if hib else float("inf"))
                   for k, hib in METRIC_HIGHER_BETTER.items()}

    def checkpoint():
        return {"model": model.net.state_dict(),
                "ema":   ema_model.net.state_dict(),
                "args":  vars(args),
                "identity_baseline": baseline}

    # ── training loop ────────────────────────────────────────────────────────
    for epoch in range(1, args.epochs + 1):
        lr = lr_at(epoch, args)
        for g in opt.param_groups:
            g["lr"] = lr

        model.train()
        tr_loss = 0.0
        for x, y in train_dl:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                enabled=args.amp):
                loss = model.flow_loss(x_post=y, x_pre=x)
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.net.parameters(), args.grad_clip)
            opt.step()
            opt.zero_grad(set_to_none=True)
            ema.update(ema_model.net, model.net)
            tr_loss += loss.item()
        tr_loss /= max(len(train_dl), 1)

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
        m = (evaluate(ema_model, val_dl, device, args, val_steps) if do_metrics
             else {k: float("nan") for k in METRIC_HIGHER_BETTER})

        if do_metrics:
            flags = "".join(
                ("+" if ((m[k] > baseline[k]) if hib else (m[k] < baseline[k])) else "-")
                for k, hib in METRIC_HIGHER_BETTER.items())
            print(f"Epoch {epoch:4d}/{args.epochs}  lr={lr:.2e}  "
                  f"train={tr_loss:.4f}  val={val_loss:.4f}  "
                  f"MAE={m['mae']:.4f}  MSE={m['mse']:.4f}  "
                  f"PSNR={m['psnr']:.2f}  SSIM={m['ssim']:.4f}  "
                  f"[vs identity: {flags}]")
        else:
            print(f"Epoch {epoch:4d}/{args.epochs}  lr={lr:.2e}  "
                  f"train={tr_loss:.4f}  val={val_loss:.4f}")

        csv_log.append({"epoch": epoch, "lr": lr, "train_loss": tr_loss,
                        "val_loss": val_loss, **m})
        log = {"train_loss": tr_loss, "val_loss": val_loss, "lr": lr}
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
    plot_progression(csv_log, out_dir, f"{out_dir.name} — Flow3D ({args.source})",
                     baseline=baseline)
    print(f"\nDone. Checkpoints in {out_dir}")
    wandb_finish(run)


if __name__ == "__main__":
    main()
