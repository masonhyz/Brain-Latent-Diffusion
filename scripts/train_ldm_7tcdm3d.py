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
import os
import random
import re
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
            # metrics are computed only every metric_every epochs, so drop the NaN
            # gaps before plotting/smoothing (train/val loss remain dense above).
            valid = ~np.isnan(s2[col])
            ep_v, val_v = s2["epoch"][valid], s2[col][valid]
            if len(val_v) == 0:
                ax.set_title(f"Stage 2 — {label}\n(no metric epochs yet)")
                ax.set_xlabel("Epoch")
                ax.set_ylabel(label)
                ax.grid(True, alpha=0.3)
                continue
            wv = max(1, len(ep_v) // 50)
            ax.plot(ep_v, val_v, color=color, linewidth=0.7, alpha=0.3)
            s = _smooth(val_v, wv)
            ax.plot(ep_v, s, color=color, linewidth=2.0)
            best_idx = int(np.nanargmax(s) if higher_better else np.nanargmin(s))
            ax.axvline(ep_v[best_idx], color="gray", linestyle=":", linewidth=1.0)
            ax.set_title(f"Stage 2 — {label}\nbest {s[best_idx]:.4f} @ ep {int(ep_v[best_idx])}")
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
            "n_folds": args.n_folds,
            "fold": args.fold,
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
    p.add_argument("--val_frac",    type=float, default=0.15,
                   help="Holdout validation fraction (ignored when --fold is set)")
    p.add_argument("--num_workers", type=int,   default=4)
    p.add_argument("--seed",        type=int,   default=42)
    # k-fold cross-validation: pass --fold to hold out one of --n_folds disjoint
    # folds instead of a --val_frac holdout. Stage 2 must use the SAME seed +
    # n_folds + fold as its stage-1 AE so both stages share the held-out subjects.
    p.add_argument("--n_folds",     type=int,   default=7,
                   help="Number of CV folds (used only when --fold is set)")
    p.add_argument("--fold",        type=int,   default=None,
                   help="Held-out fold index 0..n_folds-1. Unset = --val_frac holdout.")
    # training
    p.add_argument("--epochs",      type=int,   default=None,
                   help="Override epochs (default: 200 for stage 1, 2000 for stage 2)")
    p.add_argument("--batch_size",  type=int,   default=2)
    p.add_argument("--lr",          type=float, default=3e-4)
    p.add_argument("--amp",    dest="amp", action="store_true")
    p.add_argument("--no-amp", dest="amp", action="store_false")
    p.set_defaults(amp=True)
    # ── speed / validation cadence (all tunable) ─────────────────────────────
    # Stage-2 validation runs full DDIM generation, which is ~ddim_steps forward
    # passes per sample and dominates wall-clock. val_loss (cheap, one forward)
    # is still computed every epoch; the expensive generation + MAE/MSE/PSNR/SSIM
    # metrics run only every --metric_every epochs.
    p.add_argument("--metric_every", type=int, default=5,
                   help="Stage 2: run full generation + metrics every N epochs "
                        "(1 = every epoch). val_loss is always logged every epoch.")
    p.add_argument("--val_ddim_steps", type=int, default=None,
                   help="DDIM steps for validation-metric generation "
                        "(default: --ddim_steps). Lower = faster monitoring.")
    p.add_argument("--tf32",    dest="tf32", action="store_true",
                   help="Enable TF32 matmul/cudnn (faster on Ampere+; on by default)")
    p.add_argument("--no-tf32", dest="tf32", action="store_false")
    p.set_defaults(tf32=True)
    p.add_argument("--persistent_workers",    dest="persistent_workers",
                   action="store_true",
                   help="Keep DataLoader workers alive across epochs "
                        "(faster for many short epochs; on by default)")
    p.add_argument("--no-persistent-workers", dest="persistent_workers",
                   action="store_false")
    p.set_defaults(persistent_workers=True)
    # stage-1 early stopping: after a warmup of --es_min_epochs, stop once val
    # loss has failed to beat the running best for --es_patience epochs in a row.
    p.add_argument("--es_min_epochs", type=int, default=50,
                   help="Stage-1 early stopping: never stop before this epoch")
    p.add_argument("--es_patience",   type=int, default=10,
                   help="Stage-1 early stopping: stop after this many consecutive "
                        "epochs with no val-loss improvement")
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
    # Weights & Biases
    p.add_argument("--wandb",    dest="wandb", action="store_true")
    p.add_argument("--no-wandb", dest="wandb", action="store_false")
    p.set_defaults(wandb=True)
    p.add_argument("--wandb_project", type=str, default="moyamoya-7tcdm3d")
    p.add_argument("--wandb_entity",  type=str, default=None)
    p.add_argument("--wandb_group",   type=str, default=None,
                   help="Group all runs (folds + AE) of one experiment; "
                        "default: derived from out_dir by stripping _foldN/_ae")
    p.add_argument("--wandb_mode",    type=str, default="online",
                   choices=["online", "offline", "disabled"],
                   help="online falls back to offline automatically if not logged in")
    return p.parse_args()


# ── Weights & Biases ─────────────────────────────────────────────────────────
# Robust by design: W&B must never block or crash a training run. If it's not
# installed, or online mode is requested without credentials, we print a one-line
# notice and training proceeds (falling back to offline logging where possible).

def _has_wandb_creds() -> bool:
    if os.environ.get("WANDB_API_KEY"):
        return True
    netrc = Path.home() / ".netrc"
    try:
        return netrc.is_file() and "api.wandb.ai" in netrc.read_text()
    except Exception:
        return False


def init_wandb(args, out_dir: Path, stage: int, n_train: int, n_val: int):
    """Start a W&B run for this (fold, stage), or return None if off/unavailable.

    One run per (fold, stage); all runs of an experiment share ``group`` so the
    k folds aggregate in the UI. ``job_type`` = stageN, and fold/stage land in
    the config + tags for filtering.
    """
    if not args.wandb:
        return None
    try:
        import wandb
    except Exception:
        print("[wandb] not installed — skipping (`pip install wandb`). Training continues.")
        return None

    mode = args.wandb_mode
    if mode == "online" and not _has_wandb_creds():
        print("[wandb] online requested but not logged in → logging OFFLINE instead. "
              "Run `wandb login` once (then `wandb sync <run dir>` to upload, or just "
              "re-launch after login). Training continues.")
        mode = "offline"

    # Group all runs of one experiment. Prefer an explicit --wandb_group (the
    # shell always passes one); otherwise derive a stem from the dir name. Nested
    # k-fold dirs are just ``foldN`` — use the ``..._cv`` parent's name instead.
    if args.wandb_group:
        group = args.wandb_group
    else:
        name  = out_dir.parent.name if re.fullmatch(r"fold\d+", out_dir.name) else out_dir.name
        group = re.sub(r"_(stage1_ae|stage1|ae|fold\d+)$|_stage2_.*$|_cv$", "", name)
    fold  = getattr(args, "fold", None)
    config = dict(vars(args))
    config.update(stage=stage, n_train=n_train, n_val=n_val,
                  fold=fold, n_folds=getattr(args, "n_folds", None))
    tags = [f"stage{stage}", "kfold" if fold is not None else "holdout"]
    if fold is not None:
        tags.append(f"fold{fold}")

    try:
        run = wandb.init(
            project=args.wandb_project, entity=args.wandb_entity,
            name=f"{out_dir.name}-stage{stage}", group=group,
            job_type=f"stage{stage}", tags=tags, config=config,
            dir=str(out_dir), mode=mode, reinit=True,
        )
        print(f"[wandb] project '{args.wandb_project}' | group '{group}' | "
              f"run '{run.name}' | mode {mode}")
        return run
    except Exception as e:
        print(f"[wandb] init failed ({e}); continuing without W&B.")
        return None


def _wandb_log(run, data: dict, step: int) -> None:
    if run is not None:
        try:
            run.log(data, step=step)
        except Exception:
            pass


def _wandb_log_image(run, key: str, path, step: int) -> None:
    if run is None:
        return
    try:
        import wandb
        run.log({key: wandb.Image(str(path))}, step=step)
    except Exception:
        pass


def _wandb_finish(run) -> None:
    if run is not None:
        try:
            run.finish()
        except Exception:
            pass


# ── data ─────────────────────────────────────────────────────────────────────

def make_loaders(args):
    return build_loaders(args, augment=True)


# ── stage 1: autoencoder ─────────────────────────────────────────────────────

def train_stage1(args, device):
    # Default to a fresh timestamped run dir so a new kickoff never overwrites
    # an old one; pass --out_dir to reuse/resume a specific directory.
    out_dir = Path(args.out_dir or f"runs/ldm_7tcdm3d_{datetime.now():%Y-%m-%d_%H-%M-%S}")
    out_dir.mkdir(parents=True, exist_ok=True)
    args.out_dir = str(out_dir)
    print(f"[Stage 1] logging to {out_dir}")
    install_run_logger(out_dir, stage=1)

    train_dl, val_dl = make_loaders(args)
    save_hparams(args, len(train_dl.dataset), len(val_dl.dataset), out_dir)
    run = init_wandb(args, out_dir, 1, len(train_dl.dataset), len(val_dl.dataset))

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
    epochs_no_improve = 0

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
        _wandb_log(run, {"train_loss": tr_loss, "val_loss": val_loss}, step=epoch)

        ckpt = {"ae": ae.state_dict(), "args": vars(args)}
        torch.save(ckpt, out_dir / "stage1_last.pt")
        if val_loss < best_val:
            best_val = val_loss
            epochs_no_improve = 0
            torch.save(ckpt, out_dir / "stage1_best.pt")
            print(f"  → new best AE ({best_val:.4f})")
        else:
            epochs_no_improve += 1
            # Only arm early stopping after a warmup; the AE is cheap and
            # converges fast, so there's no need to burn the full schedule.
            if epoch > args.es_min_epochs and epochs_no_improve >= args.es_patience:
                print(f"  → early stop at epoch {epoch}: no val-loss improvement "
                      f"for {epochs_no_improve} epochs (best={best_val:.4f})")
                break

    print(f"Stage 1 done. Best checkpoint: {out_dir / 'stage1_best.pt'}")
    plot_progression(out_dir, stage=1)
    _wandb_finish(run)
    return out_dir / "stage1_best.pt"


# ── stage 2: latent diffusion ─────────────────────────────────────────────────

def _default_stage2_out_dir(ae_ckpt: str, fold) -> Path:
    """A fresh stage-2 run dir that is a SIBLING of the AE's dir (never inside it).

    Stage 1 is rarely retrained, so stage 2 gets its own timestamped folder rather
    than dumping next to the frozen AE checkpoint. An experiment stem is derived
    from the AE dir name (stripping a trailing ``_stage1_ae`` / ``_ae`` /
    ``_stage1``); the run lands at ``<stem>_stage2_<timestamp>``. A k-fold run
    (``fold`` set) nests as ``<stem>_stage2_<timestamp>_cv/fold<N>`` — but real
    k-fold sweeps pass an explicit ``--out_dir`` per fold (via train_7tcdm3d.sh)
    so all folds share one parent.
    """
    ae_dir = Path(ae_ckpt).resolve().parent
    stem   = re.sub(r"_(stage1_ae|stage1|ae)$", "", ae_dir.name)
    ts     = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    runs   = ae_dir.parent                       # e.g. runs/
    if fold is not None:
        return runs / f"{stem}_stage2_{ts}_cv" / f"fold{fold}"
    return runs / f"{stem}_stage2_{ts}"


def train_stage2(args, device):
    if args.ae_ckpt is None:
        raise ValueError("--ae_ckpt is required for stage 2")

    # Stage 2 lands in its OWN folder, separate from the (rarely retrained) AE.
    # Pass --out_dir to override; otherwise a fresh sibling dir is created.
    out_dir = (Path(args.out_dir) if args.out_dir
               else _default_stage2_out_dir(args.ae_ckpt, getattr(args, "fold", None)))
    out_dir.mkdir(parents=True, exist_ok=True)
    args.out_dir = str(out_dir)
    print(f"[Stage 2] logging to {out_dir}")
    install_run_logger(out_dir, stage=2)
    vis_dir = out_dir / "vis_stage2"
    vis_dir.mkdir(exist_ok=True)

    train_dl, val_dl = make_loaders(args)
    save_hparams(args, len(train_dl.dataset), len(val_dl.dataset), out_dir)
    run = init_wandb(args, out_dir, 2, len(train_dl.dataset), len(val_dl.dataset))

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

    # Full generation dominates wall-clock, so it runs only every metric_every
    # epochs (val_loss stays every epoch). val_ddim_steps defaults to ddim_steps.
    val_ddim_steps = args.val_ddim_steps or args.ddim_steps
    print(f"  Metrics every {args.metric_every} epoch(s) with {val_ddim_steps} DDIM steps")

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
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
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

        # ── validation ────────────────────────────────────────────────────────
        # val_loss (one forward) every epoch; the expensive full-generation
        # metrics only every metric_every epochs (and always on the final epoch).
        do_metrics = (args.metric_every <= 1
                      or epoch % args.metric_every == 0
                      or epoch == args.epochs)
        model.denoiser.eval()
        val_loss = 0.0
        all_metrics = {"mae": [], "mse": [], "psnr": [], "ssim": []}
        with torch.no_grad():
            for x, y in val_dl:
                x_d, y_d = (x.to(device, non_blocking=True),
                            y.to(device, non_blocking=True))
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=args.amp):
                    val_loss += model.p_loss(x_post=y_d, x_pre=x_d).item()
                    if do_metrics:
                        pred = model.generate(x_d, ddim_steps=val_ddim_steps,
                                              guidance_scale=args.guidance_scale)
                if do_metrics:
                    mask = (x != 0) | (y != 0)
                    m = compute_metrics(pred.float().cpu(), y, mask)
                    for k in all_metrics:
                        all_metrics[k].append(m[k])

        val_loss /= len(val_dl)
        epoch_metrics = ({k: float(np.mean(v)) for k, v in all_metrics.items()}
                         if do_metrics
                         else {k: float("nan") for k in all_metrics})

        if do_metrics:
            print(f"Epoch {epoch:4d}/{args.epochs}  "
                  f"train={tr_loss:.4f}  val={val_loss:.4f}  "
                  f"MAE={epoch_metrics['mae']:.4f}  MSE={epoch_metrics['mse']:.4f}  "
                  f"PSNR={epoch_metrics['psnr']:.2f}  SSIM={epoch_metrics['ssim']:.4f}")
        else:
            print(f"Epoch {epoch:4d}/{args.epochs}  "
                  f"train={tr_loss:.4f}  val={val_loss:.4f}  (metrics skipped)")

        _append_metrics_row(csv_path, epoch, tr_loss, val_loss, epoch_metrics)
        log_data = {"train_loss": tr_loss, "val_loss": val_loss}
        if do_metrics:
            log_data.update(epoch_metrics)
        _wandb_log(run, log_data, step=epoch)

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
        # (only on epochs where metrics were actually computed)
        if do_metrics:
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
            _wandb_log_image(run, "val_sample", vis_dir / f"epoch_{epoch:04d}.png", step=epoch)

    print(f"Stage 2 done. Best checkpoint: {out_dir / 'stage2_best.pt'}")
    plot_progression(out_dir, stage=2)
    _wandb_finish(run)


# ── entry point ───────────────────────────────────────────────────────────────

def main():
    args = get_args()
    if args.epochs is None:
        args.epochs = 200 if args.stage == 1 else 2000
    seed_everything(args.seed)
    if args.tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    device = get_device()
    print(f"Device: {device}")

    if args.stage == 1:
        train_stage1(args, device)
    else:
        train_stage2(args, device)


if __name__ == "__main__":
    main()
