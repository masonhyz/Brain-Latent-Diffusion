"""
CDM3D training — single-stage, image-space conditional diffusion.

No autoencoder. The denoiser operates directly on the full-resolution MRI volume.
EMA weights are maintained and saved alongside the online model.

Usage:
    python scripts/train_cdm3d.py --data_root fmri --out_dir runs/cdm3d
"""

import argparse
import copy
import csv
import json
import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))

from moyamoya.data import build_loaders
from moyamoya.utils import seed_everything, get_device
from moyamoya.metrics import compute_metrics as _compute_metrics
from moyamoya.viz import percentile_norm as _to_np
from moyamoya.models.cdm3d import build_cdm3d, EMA


# ── visualisation ─────────────────────────────────────────────────────────────

def _get_slices(vol_np: np.ndarray) -> dict:
    D, H, W = vol_np.shape
    return {
        "Axial":    vol_np[D // 2, :, :],
        "Coronal":  vol_np[:, H // 2, :],
        "Sagittal": vol_np[:, :, W // 2],
    }


def save_grid(
    x_np: np.ndarray,
    y_np: np.ndarray,
    pred_np: np.ndarray,
    title: str,
    save_path: Path,
) -> None:
    xs = _get_slices(x_np)
    ys = _get_slices(y_np)
    ps = _get_slices(pred_np)

    views = ["Axial", "Coronal", "Sagittal"]
    cols  = ["Pre-surgery", "Prediction", "Post-surgery GT"]

    fig, axes = plt.subplots(3, 3, figsize=(10, 10), constrained_layout=True)
    for r, view in enumerate(views):
        for c, (img, col_title) in enumerate(zip(
            [xs[view], ps[view], ys[view]], cols
        )):
            ax = axes[r, c]
            ax.imshow(img, cmap="gray", origin="lower", vmin=0, vmax=1)
            ax.axis("off")
            if r == 0:
                ax.set_title(col_title, fontsize=11)
            if c == 0:
                ax.set_ylabel(view, fontsize=10)

    fig.suptitle(title, fontsize=12)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> dict:
    # Sampled predictions during training can be unstable → sanitise NaN/Inf.
    return _compute_metrics(pred, target, mask, sanitize_pred=True)


# ── training progression plot ─────────────────────────────────────────────────

def _smooth(arr: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return arr
    return np.convolve(arr, np.ones(window) / window, mode="same")


def _load_metrics_csv(path: Path) -> dict:
    rows = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    if not rows:
        return {}
    cols = {k: [] for k in rows[0]}
    for row in rows:
        for k, v in row.items():
            try:
                cols[k].append(float(v))
            except (ValueError, TypeError):
                cols[k].append(float("nan"))
    return {k: np.array(v) for k, v in cols.items()}


def plot_progression(out_dir: Path) -> None:
    csv_path = out_dir / "metrics.csv"
    if not csv_path.exists():
        return

    import matplotlib.gridspec as gridspec

    s = _load_metrics_csv(csv_path)
    w = max(1, len(s["epoch"]) // 50)

    fig = plt.figure(figsize=(16, 8))
    fig.suptitle(f"{out_dir.name} — CDM3D Training", fontsize=14, fontweight="bold", y=0.98)
    gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.42, wspace=0.35)

    ax0 = fig.add_subplot(gs[0, :2])
    ax0.plot(s["epoch"], s["train_loss"], alpha=0.35, linewidth=0.8, color="tab:blue")
    ax0.plot(s["epoch"], s["val_loss"],   alpha=0.35, linewidth=0.8, color="tab:orange")
    ax0.plot(s["epoch"], _smooth(s["train_loss"], w),
             label="Train (smooth)", linewidth=1.8, color="tab:blue")
    ax0.plot(s["epoch"], _smooth(s["val_loss"], w),
             label="Val (smooth)", linewidth=1.8, color="tab:orange", linestyle="--")
    ax0.set_title("Diffusion Loss")
    ax0.set_xlabel("Epoch")
    ax0.set_ylabel("Loss")
    ax0.legend()
    ax0.grid(True, alpha=0.3)

    metrics_cfg = [
        ("mae",  "MAE",       "tab:red",    False),
        ("mse",  "MSE",       "tab:brown",  False),
        ("psnr", "PSNR (dB)", "tab:green",  True),
        ("ssim", "SSIM",      "tab:purple", True),
    ]
    for i, (col, label, color, higher_better) in enumerate(metrics_cfg):
        ax = fig.add_subplot(gs[1, i])
        ax.plot(s["epoch"], s[col], color=color, linewidth=0.7, alpha=0.3)
        sm = _smooth(s[col], w)
        ax.plot(s["epoch"], sm, color=color, linewidth=2.0)
        best_idx = int(np.nanargmax(sm) if higher_better else np.nanargmin(sm))
        ax.axvline(s["epoch"][best_idx], color="gray", linestyle=":", linewidth=1.0)
        ax.set_title(f"{label}\nbest {sm[best_idx]:.4f} @ ep {int(s['epoch'][best_idx])}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.3)

    out = out_dir / "training_progression.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → training progression saved: {out}")


# ── args ─────────────────────────────────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root",       type=str,   default="fmri")
    p.add_argument("--out_dir",         type=str,   default=None)
    p.add_argument("--resume",          type=str,   default=None,
                   help="Path to a checkpoint to resume from")
    # data
    p.add_argument("--val_frac",        type=float, default=0.15)
    p.add_argument("--num_workers",     type=int,   default=4)
    p.add_argument("--seed",            type=int,   default=42)
    # training
    p.add_argument("--epochs",          type=int,   default=2000)
    p.add_argument("--batch_size",      type=int,   default=1)
    p.add_argument("--lr",              type=float, default=1e-4)
    p.add_argument("--ema_beta",        type=float, default=0.999)
    p.add_argument("--grad_clip",       type=float, default=1.0)
    p.add_argument("--amp",    dest="amp", action="store_true")
    p.add_argument("--no-amp", dest="amp", action="store_false")
    p.set_defaults(amp=True)
    # architecture
    p.add_argument("--dim",             type=int,   default=32)
    p.add_argument("--dim_mults",       type=int,   nargs="+", default=[1, 2, 4, 8])
    p.add_argument("--init_kernel_size",type=int,   default=3,
                   help="7 for full-res on large GPUs; 3 for <12 GB")
    p.add_argument("--resnet_groups",   type=int,   default=8)
    # diffusion
    p.add_argument("--T",               type=int,   default=1000)
    p.add_argument("--ddim_steps",      type=int,   default=50)
    p.add_argument("--ssim_weight",     type=float, default=0.5)
    p.add_argument("--cfg_drop_prob",   type=float, default=0.15)
    p.add_argument("--guidance_scale",  type=float, default=3.0)
    p.add_argument("--vis_every",       type=int,   default=10)
    return p.parse_args()


# ── data ─────────────────────────────────────────────────────────────────────

def make_loaders(args):
    return build_loaders(args, augment=True)


# ── checkpoint helpers ────────────────────────────────────────────────────────

def save_ckpt(path: Path, model, ema_model, opt, epoch: int, args) -> None:
    torch.save({
        "model":     model.state_dict(),
        "ema_model": ema_model.state_dict(),
        "opt":       opt.state_dict(),
        "epoch":     epoch,
        "args":      vars(args),
    }, path)


def load_ckpt(path: Path, model, ema_model, opt, device):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"])
    ema_model.load_state_dict(ckpt["ema_model"])
    opt.load_state_dict(ckpt["opt"])
    return ckpt["epoch"]


# ── main training loop ────────────────────────────────────────────────────────

def main():
    args   = get_args()
    seed_everything(args.seed)
    device = get_device()
    print(f"Device: {device}")

    out_dir = Path(args.out_dir or "runs/cdm3d")
    out_dir.mkdir(parents=True, exist_ok=True)
    vis_dir = out_dir / "vis"
    vis_dir.mkdir(exist_ok=True)

    train_dl, val_dl = make_loaders(args)

    # save hyperparameters
    with open(out_dir / "hparams.json", "w") as f:
        json.dump({
            "model": "cdm3d",
            "timestamp": datetime.now().isoformat(),
            "args": vars(args),
            "dataset": {
                "data_root": args.data_root,
                "n_train": len(train_dl.dataset),
                "n_val": len(val_dl.dataset),
            },
        }, f, indent=2)

    model = build_cdm3d(
        dim=args.dim,
        dim_mults=tuple(args.dim_mults),
        init_kernel_size=args.init_kernel_size,
        resnet_groups=args.resnet_groups,
        timesteps=args.T,
        ddim_steps=args.ddim_steps,
        ssim_weight=args.ssim_weight,
        cfg_drop_prob=args.cfg_drop_prob,
    ).to(device)

    ema_model = copy.deepcopy(model).to(device)
    ema_model.eval()
    ema = EMA(beta=args.ema_beta)

    opt    = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[CDM3D] params: {n_params:,}")
    print(f"  Train {len(train_dl.dataset)} / Val {len(val_dl.dataset)} samples")

    start_epoch = 1
    if args.resume:
        start_epoch = load_ckpt(Path(args.resume), model, ema_model, opt, device) + 1
        print(f"  Resumed from {args.resume} (epoch {start_epoch - 1})")

    csv_path = out_dir / "metrics.csv"
    if start_epoch == 1:
        with open(csv_path, "w", newline="") as f:
            csv.writer(f).writerow(
                ["epoch", "train_loss", "val_loss", "mae", "mse", "psnr", "ssim"]
            )

    best_val = float("inf")

    for epoch in range(start_epoch, args.epochs + 1):
        # ── train ─────────────────────────────────────────────────────────────
        model.train()
        tr_loss = 0.0
        n_steps = 0
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            with torch.cuda.amp.autocast(enabled=args.amp):
                loss = model(x_post=y, x_pre=x)
            # A non-finite loss would push inf/nan into the weights (and the EMA
            # model used for sampling), turning every later metric into nan.
            # Drop the batch instead of corrupting the run.
            if not torch.isfinite(loss):
                opt.zero_grad(set_to_none=True)
                continue
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            tr_loss += loss.item()
            n_steps += 1
            ema.update(ema_model.denoise_fn, model.denoise_fn)
        tr_loss /= max(n_steps, 1)

        # ── validate ──────────────────────────────────────────────────────────
        ema_model.eval()
        val_loss    = 0.0
        all_metrics = {"mae": [], "mse": [], "psnr": [], "ssim": []}

        with torch.no_grad():
            for x, y in val_dl:
                x_d, y_d = x.to(device), y.to(device)
                with torch.cuda.amp.autocast(enabled=args.amp):
                    val_loss += ema_model(x_post=y_d, x_pre=x_d).item()
                # sample() forces fp32 internally — keep it out of the autocast
                # block so the DDIM loop never accumulates in fp16.
                pred = ema_model.sample(x_d, guidance_scale=args.guidance_scale)

                mask = (x != 0) | (y != 0)
                m = compute_metrics(pred.float().cpu(), y, mask)
                for k in all_metrics:
                    all_metrics[k].append(m[k])

        val_loss /= len(val_dl)
        epoch_metrics = {k: float(np.mean(v)) for k, v in all_metrics.items()}

        print(
            f"Epoch {epoch:4d}/{args.epochs}  "
            f"train={tr_loss:.4f}  val={val_loss:.4f}  "
            f"MAE={epoch_metrics['mae']:.4f}  MSE={epoch_metrics['mse']:.4f}  "
            f"PSNR={epoch_metrics['psnr']:.2f}  SSIM={epoch_metrics['ssim']:.4f}"
        )

        with open(csv_path, "a", newline="") as f:
            csv.writer(f).writerow([
                epoch, f"{tr_loss:.6f}", f"{val_loss:.6f}",
                f"{epoch_metrics['mae']:.6f}", f"{epoch_metrics['mse']:.6f}",
                f"{epoch_metrics['psnr']:.4f}", f"{epoch_metrics['ssim']:.6f}",
            ])

        save_ckpt(out_dir / "last.pt", model, ema_model, opt, epoch, args)
        if val_loss < best_val:
            best_val = val_loss
            save_ckpt(out_dir / "best.pt", model, ema_model, opt, epoch, args)
            print(f"  → new best ({best_val:.4f})")

        # ── visualisation ─────────────────────────────────────────────────────
        if args.vis_every > 0 and epoch % args.vis_every == 0:
            val_dataset = val_dl.dataset
            rand_idx = random.randint(0, len(val_dataset) - 1)
            x_vis, y_vis = val_dataset[rand_idx]
            x_b = x_vis.unsqueeze(0).to(device)
            y_b = y_vis.unsqueeze(0).to(device)
            with torch.no_grad():
                pred_b = ema_model.sample(x_b, guidance_scale=args.guidance_scale)
            save_grid(
                _to_np(x_b), _to_np(y_b), _to_np(pred_b),
                f"CDM3D  Epoch {epoch}",
                vis_dir / f"epoch_{epoch:04d}.png",
            )
            print(f"  → saved visualisation: {vis_dir / f'epoch_{epoch:04d}.png'}")

    print(f"Done. Best checkpoint: {out_dir / 'best.pt'}")
    plot_progression(out_dir)


if __name__ == "__main__":
    main()
