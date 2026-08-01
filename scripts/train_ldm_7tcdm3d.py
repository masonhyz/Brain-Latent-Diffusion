"""
PairedLatentDiffusion training with 7TCDM-style 3-D denoiser — two stages.

Stage 1: train the KL-autoencoder to reconstruct fMRI volumes.
Stage 2: freeze the AE; train the 7TCDM-derived 3-D latent diffusion U-Net.

Usage:
    # Stage 1
    python scripts/train_ldm_7tcdm3d.py --stage 1 --out_dir runs/ldm_7tcdm3d

    # Stage 2 (requires a Stage-1 checkpoint)
    python scripts/train_ldm_7tcdm3d.py --stage 2 \\
        --ae_ckpt runs/ldm_7tcdm3d/stage1_best.pt \\
        --out_dir runs/ldm_7tcdm3d
"""

import argparse
import csv
import json
import random
import subprocess
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
from moyamoya.metrics import compute_metrics
from moyamoya.models.ldm_7tcdm3d import build_paired_latent_diffusion_7tcdm
from moyamoya.models.ldm3d import build_autoencoder


# ── run logging: mirror the console to a per-run logfile ─────────────────────

class _Tee:
    """Duplicate stream writes to the real console and a logfile."""

    def __init__(self, *streams):
        self._streams = streams

    def write(self, data):
        for s in self._streams:
            s.write(data)
            s.flush()

    def flush(self):
        for s in self._streams:
            s.flush()


def install_run_logger(out_dir: Path, stage: int) -> Path:
    """Tee stdout/stderr into out_dir/train_stage{stage}.log (append mode).

    Runs are appended (not truncated) so re-running a stage into the same
    out_dir preserves earlier console history, each delimited by a banner.
    """
    log_path = out_dir / f"train_stage{stage}.log"
    logfile = open(log_path, "a", buffering=1)  # line-buffered
    logfile.write(f"\n===== stage {stage} run started {datetime.now().isoformat()} =====\n")
    sys.stdout = _Tee(sys.__stdout__, logfile)
    sys.stderr = _Tee(sys.__stderr__, logfile)
    return log_path


def _git_commit():
    """Short git SHA of the working tree, or None if unavailable."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).resolve().parent.parent,
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return None


# ── visualisation helpers (matching eval_ldm_7tcdm3d.py) ─────────────────────

from moyamoya.viz import percentile_norm as _to_np


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
    """3×3 grid: (axial/coronal/sagittal) × (pre | prediction | post GT)."""
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




# ── training progression plot ─────────────────────────────────────────────────

def _smooth(arr: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return arr
    return np.convolve(arr, np.ones(window) / window, mode="same")


def _load_metrics_csv(path: Path) -> dict:
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    if not rows:
        return {}
    cols: dict = {k: [] for k in rows[0]}
    for row in rows:
        for k, v in row.items():
            try:
                cols[k].append(float(v))
            except (ValueError, TypeError):
                cols[k].append(float("nan"))
    return {k: np.array(v) for k, v in cols.items()}


def plot_progression(out_dir: Path, stage: int) -> None:
    import matplotlib.gridspec as gridspec

    if stage == 1:
        csv1 = out_dir / "metrics_stage1.csv"
        if not csv1.exists():
            return
        s1 = _load_metrics_csv(csv1)
        mask = ~np.isnan(s1["train_loss"])
        s1 = {k: v[mask] for k, v in s1.items()}

        fig, ax = plt.subplots(figsize=(7, 4))
        fig.suptitle(f"{out_dir.name} — Stage 1: Autoencoder Loss",
                     fontsize=12, fontweight="bold")
        ax.plot(s1["epoch"], s1["train_loss"], label="Train", linewidth=1.8)
        ax.plot(s1["epoch"], s1["val_loss"],   label="Val",   linewidth=1.8, linestyle="--")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend()
        ax.grid(True, alpha=0.3)
        out = out_dir / "training_progression_stage1.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  → training progression saved: {out}")

    elif stage == 2:
        csv1 = out_dir / "metrics_stage1.csv"
        csv2 = out_dir / "metrics_stage2.csv"
        if not csv2.exists():
            return

        has_s1 = csv1.exists()
        n_rows = 2
        fig = plt.figure(figsize=(16, 10))
        fig.suptitle(f"{out_dir.name} — Training Progression",
                     fontsize=14, fontweight="bold", y=0.98)

        top_cols = 4
        gs = gridspec.GridSpec(2, top_cols, figure=fig, hspace=0.42, wspace=0.35)

        s2 = _load_metrics_csv(csv2)
        w = max(1, len(s2["epoch"]) // 50)

        col_start = 0
        if has_s1:
            s1 = _load_metrics_csv(csv1)
            mask = ~np.isnan(s1["train_loss"])
            s1 = {k: v[mask] for k, v in s1.items()}
            ax0 = fig.add_subplot(gs[0, :2])
            ax0.plot(s1["epoch"], s1["train_loss"], label="Train", linewidth=1.8)
            ax0.plot(s1["epoch"], s1["val_loss"],   label="Val",   linewidth=1.8, linestyle="--")
            ax0.set_title("Stage 1 — Autoencoder Loss")
            ax0.set_xlabel("Epoch")
            ax0.set_ylabel("Loss")
            ax0.legend()
            ax0.grid(True, alpha=0.3)
            col_start = 2

        ax1 = fig.add_subplot(gs[0, col_start:])
        ax1.plot(s2["epoch"], s2["train_loss"], alpha=0.35, linewidth=0.8, color="tab:blue")
        ax1.plot(s2["epoch"], s2["val_loss"],   alpha=0.35, linewidth=0.8, color="tab:orange")
        ax1.plot(s2["epoch"], _smooth(s2["train_loss"], w),
                 label="Train (smooth)", linewidth=1.8, color="tab:blue")
        ax1.plot(s2["epoch"], _smooth(s2["val_loss"], w),
                 label="Val (smooth)", linewidth=1.8, color="tab:orange", linestyle="--")
        ax1.set_title("Stage 2 — Diffusion Loss")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        metrics_cfg = [
            ("mae",  "MAE",       "tab:red",    False),
            ("mse",  "MSE",       "tab:brown",  False),
            ("psnr", "PSNR (dB)", "tab:green",  True),
            ("ssim", "SSIM",      "tab:purple", True),
        ]
        for i, (col, label, color, higher_better) in enumerate(metrics_cfg):
            ax = fig.add_subplot(gs[1, i])
            ax.plot(s2["epoch"], s2[col], color=color, linewidth=0.7, alpha=0.3)
            s = _smooth(s2[col], w)
            ax.plot(s2["epoch"], s, color=color, linewidth=2.0)
            best_idx = int(np.nanargmax(s) if higher_better else np.nanargmin(s))
            ax.axvline(s2["epoch"][best_idx], color="gray", linestyle=":", linewidth=1.0)
            ax.set_title(f"Stage 2 — {label}\nbest {s[best_idx]:.4f} @ ep {int(s2['epoch'][best_idx])}")
            ax.set_xlabel("Epoch")
            ax.set_ylabel(label)
            ax.grid(True, alpha=0.3)

        out = out_dir / "training_progression.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  → training progression saved: {out}")


# ── hyperparameter logging ────────────────────────────────────────────────────

def save_hparams(args, n_train: int, n_val: int, out_dir: Path) -> None:
    hparams = {
        "model": "ldm_7tcdm3d",
        "stage": args.stage,
        "timestamp": datetime.now().isoformat(),
        "git_commit": _git_commit(),
        "args": vars(args),
        "dataset": {
            "data_root": args.data_root,
            "n_train": n_train,
            "n_val": n_val,
            "val_frac": args.val_frac,
            "seed": args.seed,
        },
    }
    path = out_dir / f"hparams_stage{args.stage}.json"
    with open(path, "w") as f:
        json.dump(hparams, f, indent=2)
    print(f"  Hyperparameters saved → {path}")


def _init_metrics_csv(path: Path) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss", "mae", "mse", "psnr", "ssim"])


def _append_metrics_row(path: Path, epoch: int, tr: float, val: float, m: dict) -> None:
    with open(path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([epoch, f"{tr:.6f}", f"{val:.6f}",
                         f"{m['mae']:.6f}", f"{m['mse']:.6f}",
                         f"{m['psnr']:.4f}", f"{m['ssim']:.6f}"])


# ── args ─────────────────────────────────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--stage",       type=int,   default=1, choices=[1, 2])
    p.add_argument("--ae_ckpt",     type=str,   default=None,
                   help="Stage-1 checkpoint (required for stage 2)")
    p.add_argument("--data_root",   type=str,   default="fmri")
    p.add_argument("--out_dir",     type=str,   default=None)
    # data
    p.add_argument("--val_frac",    type=float, default=0.15)
    p.add_argument("--num_workers", type=int,   default=4)
    p.add_argument("--seed",        type=int,   default=42)
    # training
    p.add_argument("--epochs",      type=int,   default=None,
                   help="Override epochs (default: 200 for stage 1, 2000 for stage 2)")
    p.add_argument("--batch_size",  type=int,   default=2)
    p.add_argument("--lr",          type=float, default=3e-4)
    p.add_argument("--amp",    dest="amp", action="store_true")
    p.add_argument("--no-amp", dest="amp", action="store_false")
    p.set_defaults(amp=True)
    # autoencoder architecture
    p.add_argument("--z_channels",    type=int,   default=4)
    p.add_argument("--embed_dim",     type=int,   default=4)
    p.add_argument("--ae_ch",         type=int,   default=64)
    p.add_argument("--ae_ch_mult",    type=int,   nargs="+", default=[1, 2, 4],
                   help="Channel multipliers per AE encoder level (fewer = larger latent)")
    p.add_argument("--ae_res_blocks", type=int,   default=3)
    p.add_argument("--ae_resolution", type=int,   default=64)
    p.add_argument("--kl_weight",     type=float, default=1e-6)
    # 7TCDM-style 3-D denoiser architecture
    p.add_argument("--diff_dim",         type=int,   default=32,
                   help="Base channel count for the 3-D U-Net denoiser")
    p.add_argument("--diff_dim_mults",   type=int,   nargs="+", default=[1, 2, 4, 8],
                   help="Channel multipliers per encoder level")
    p.add_argument("--init_kernel_size", type=int,   default=3)
    p.add_argument("--resnet_groups",    type=int,   default=8)
    # diffusion schedule
    p.add_argument("--T",           type=int,   default=1000)
    p.add_argument("--ddim_steps",  type=int,   default=50)
    p.add_argument("--cfg_drop_prob",   type=float, default=0.15)
    p.add_argument("--guidance_scale",  type=float, default=3.0)
    p.add_argument("--vis_every",       type=int,   default=5)
    return p.parse_args()


# ── data ─────────────────────────────────────────────────────────────────────

def make_loaders(args):
    return build_loaders(args, augment=True)


# ── stage 1: autoencoder ─────────────────────────────────────────────────────

def train_stage1(args, device):
    out_dir = Path(args.out_dir or "runs/ldm_7tcdm3d")
    out_dir.mkdir(parents=True, exist_ok=True)
    install_run_logger(out_dir, stage=1)

    train_dl, val_dl = make_loaders(args)
    save_hparams(args, len(train_dl.dataset), len(val_dl.dataset), out_dir)

    ae = build_autoencoder(
        z_channels=args.z_channels,
        embed_dim=args.embed_dim,
        ch=args.ae_ch,
        ch_mult=tuple(args.ae_ch_mult),
        num_res_blocks=args.ae_res_blocks,
        resolution=args.ae_resolution,
        kl_weight=args.kl_weight,
    ).to(device)

    opt    = torch.optim.AdamW(ae.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=False)  # bf16 autocast needs no loss scaling

    print(f"[Stage 1 – AE] params: {sum(p.numel() for p in ae.parameters()):,}")
    print(f"  Train {len(train_dl.dataset)} / Val {len(val_dl.dataset)} samples")

    csv_path = out_dir / "metrics_stage1.csv"
    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "val_loss"])

    best_val = float("inf")

    for epoch in range(1, args.epochs + 1):
        ae.train()
        tr_loss = 0.0
        for x, y in train_dl:
            for vol in (x.to(device), y.to(device)):
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=args.amp):
                    loss = ae.rec_loss(vol)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(ae.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                tr_loss += loss.item()
        tr_loss /= 2 * len(train_dl)

        ae.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_dl:
                x_d, y_d = x.to(device), y.to(device)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=args.amp):
                    val_loss += ae.rec_loss(x_d).item()
                    val_loss += ae.rec_loss(y_d).item()
        val_loss /= 2 * len(val_dl)

        print(f"Epoch {epoch:4d}/{args.epochs}  train={tr_loss:.4f}  val={val_loss:.4f}")
        with open(csv_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch, f"{tr_loss:.6f}", f"{val_loss:.6f}"])

        ckpt = {"ae": ae.state_dict(), "args": vars(args)}
        torch.save(ckpt, out_dir / "stage1_last.pt")
        if val_loss < best_val:
            best_val = val_loss
            torch.save(ckpt, out_dir / "stage1_best.pt")
            print(f"  → new best AE ({best_val:.4f})")

    print(f"Stage 1 done. Best checkpoint: {out_dir / 'stage1_best.pt'}")
    plot_progression(out_dir, stage=1)
    return out_dir / "stage1_best.pt"


# ── stage 2: latent diffusion ─────────────────────────────────────────────────

def train_stage2(args, device):
    if args.ae_ckpt is None:
        raise ValueError("--ae_ckpt is required for stage 2")

    out_dir = Path(args.out_dir or "runs/ldm_7tcdm3d")
    out_dir.mkdir(parents=True, exist_ok=True)
    install_run_logger(out_dir, stage=2)
    vis_dir = out_dir / "vis_stage2"
    vis_dir.mkdir(exist_ok=True)

    train_dl, val_dl = make_loaders(args)
    save_hparams(args, len(train_dl.dataset), len(val_dl.dataset), out_dir)

    model = build_paired_latent_diffusion_7tcdm(
        z_channels=args.z_channels,
        embed_dim=args.embed_dim,
        ae_ch=args.ae_ch,
        ae_ch_mult=tuple(args.ae_ch_mult),
        ae_res_blocks=args.ae_res_blocks,
        ae_resolution=args.ae_resolution,
        kl_weight=args.kl_weight,
        diff_dim=args.diff_dim,
        diff_dim_mults=tuple(args.diff_dim_mults),
        init_kernel_size=args.init_kernel_size,
        resnet_groups=args.resnet_groups,
        T=args.T,
    ).to(device)

    ae_state = torch.load(args.ae_ckpt, map_location=device)["ae"]
    model.ae.load_state_dict(ae_state)
    for p in model.ae.parameters():
        p.requires_grad_(False)
    model.ae.eval()

    opt    = torch.optim.AdamW(model.denoiser.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=False)  # bf16 autocast needs no loss scaling

    n_denoiser = sum(p.numel() for p in model.denoiser.parameters())
    print(f"[Stage 2 – 7TCDM-3D Latent Diffusion] denoiser params: {n_denoiser:,}")
    print(f"  Train {len(train_dl.dataset)} / Val {len(val_dl.dataset)} samples")

    csv_path = out_dir / "metrics_stage2.csv"
    _init_metrics_csv(csv_path)

    best_val = float("inf")
    # running best checkpoint per evaluation metric (mae/mse: lower better, psnr/ssim: higher)
    metric_higher_better = {"mae": False, "mse": False, "psnr": True, "ssim": True}
    best_metric = {k: (float("-inf") if hib else float("inf"))
                   for k, hib in metric_higher_better.items()}

    for epoch in range(1, args.epochs + 1):
        model.denoiser.train()
        tr_loss = 0.0
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=args.amp):
                loss = model.p_loss(x_post=y, x_pre=x, cfg_drop_prob=args.cfg_drop_prob)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.denoiser.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            tr_loss += loss.item()
        tr_loss /= len(train_dl)

        # ── validation: denoising loss + full generation metrics ──────────────
        model.denoiser.eval()
        val_loss = 0.0
        all_metrics = {"mae": [], "mse": [], "psnr": [], "ssim": []}
        with torch.no_grad():
            for x, y in val_dl:
                x_d, y_d = x.to(device), y.to(device)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=args.amp):
                    val_loss += model.p_loss(x_post=y_d, x_pre=x_d).item()
                    pred = model.generate(x_d, ddim_steps=args.ddim_steps,
                                          guidance_scale=args.guidance_scale)

                mask = (x != 0) | (y != 0)
                m = compute_metrics(pred.float().cpu(), y, mask)
                for k in all_metrics:
                    all_metrics[k].append(m[k])

        val_loss /= len(val_dl)
        epoch_metrics = {k: float(np.mean(v)) for k, v in all_metrics.items()}

        print(f"Epoch {epoch:4d}/{args.epochs}  "
              f"train={tr_loss:.4f}  val={val_loss:.4f}  "
              f"MAE={epoch_metrics['mae']:.4f}  MSE={epoch_metrics['mse']:.4f}  "
              f"PSNR={epoch_metrics['psnr']:.2f}  SSIM={epoch_metrics['ssim']:.4f}")

        _append_metrics_row(csv_path, epoch, tr_loss, val_loss, epoch_metrics)

        # AE is frozen in stage 2, so it is NOT duplicated into every stage-2
        # checkpoint; it lives once in stage1_best.pt and is referenced here.
        # Loaders (moyamoya.models.ldm_7tcdm3d.load_7tcdm3d_checkpoint) resolve
        # it from this reference, or fall back to a sibling stage1_best.pt.
        ckpt = {
            "denoiser": model.denoiser.state_dict(),
            "args":     vars(args),
            "ae_ckpt":  args.ae_ckpt,
        }
        torch.save(ckpt, out_dir / "stage2_last.pt")
        if val_loss < best_val:
            best_val = val_loss
            torch.save(ckpt, out_dir / "stage2_best.pt")
            print(f"  → new best val_loss ({best_val:.4f})")
        # additionally keep the best checkpoint for each evaluation metric
        for k, hib in metric_higher_better.items():
            cur = epoch_metrics[k]
            if (cur > best_metric[k]) if hib else (cur < best_metric[k]):
                best_metric[k] = cur
                torch.save(ckpt, out_dir / f"stage2_best_{k}.pt")
                print(f"  → new best {k} ({cur:.4f})")

        # ── visualisation every vis_every epochs ─────────────────────────────
        if args.vis_every > 0 and epoch % args.vis_every == 0:
            val_dataset = val_dl.dataset
            rand_idx = random.randint(0, len(val_dataset) - 1)
            x_vis, y_vis = val_dataset[rand_idx]
            x_b = x_vis.unsqueeze(0).to(device)
            y_b = y_vis.unsqueeze(0).to(device)
            with torch.no_grad():
                pred_b = model.generate(x_b, ddim_steps=args.ddim_steps,
                                        guidance_scale=args.guidance_scale)
            save_grid(
                _to_np(x_b), _to_np(y_b), _to_np(pred_b),
                f"Stage 2 – LDM-7TCDM3D  Epoch {epoch}",
                vis_dir / f"epoch_{epoch:04d}.png",
            )
            print(f"  → saved visualisation: {vis_dir / f'epoch_{epoch:04d}.png'}")

    print(f"Stage 2 done. Best checkpoint: {out_dir / 'stage2_best.pt'}")
    plot_progression(out_dir, stage=2)


# ── entry point ───────────────────────────────────────────────────────────────

def main():
    args = get_args()
    if args.epochs is None:
        args.epochs = 200 if args.stage == 1 else 2000
    seed_everything(args.seed)
    device = get_device()
    print(f"Device: {device}")

    if args.stage == 1:
        train_stage1(args, device)
    else:
        train_stage2(args, device)


if __name__ == "__main__":
    main()
