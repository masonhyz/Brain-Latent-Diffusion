"""
PairedDiffusion training.

    python scripts/train_ldm.py --out_dir runs/paired_diff
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))

from moyamoya.data import build_loaders
from moyamoya.utils import seed_everything
from moyamoya.models.ldm import build_paired_diffusion


# ── visualisation helpers ────────────────────────────────────────────────────

from moyamoya.viz import percentile_norm as _to_np


def _mid(arr: np.ndarray, axis: int) -> int:
    return arr.shape[axis] // 2


def visualize_pred(x_pre, x_post_real, x_post_pred, epoch: int, save_path: Path):
    """3-column (pre | predicted post | real post) × 3-row (axial | coronal | sagittal)."""
    pre  = _to_np(x_pre)
    real = _to_np(x_post_real)
    pred = _to_np(x_post_pred)

    slices = {
        "Axial":    (pre[_mid(pre, 0)],      real[_mid(real, 0)],      pred[_mid(pred, 0)]),
        "Coronal":  (pre[:, _mid(pre, 1)],   real[:, _mid(real, 1)],   pred[:, _mid(pred, 1)]),
        "Sagittal": (pre[:, :, _mid(pre, 2)],real[:, :, _mid(real, 2)],pred[:, :, _mid(pred, 2)]),
    }

    fig, axes = plt.subplots(3, 3, figsize=(10, 10))
    cols = ["Pre-surgery", "Predicted Post", "Real Post"]
    for row_i, (view, (s_pre, s_real, s_pred)) in enumerate(slices.items()):
        for col_i, (ax, img, title) in enumerate(
            zip(axes[row_i], [s_pre, s_pred, s_real], cols)
        ):
            ax.imshow(img, cmap="hot", origin="lower")
            ax.axis("off")
            if row_i == 0:
                ax.set_title(title, fontsize=11)
            if col_i == 0:
                ax.set_ylabel(view, fontsize=10)

    fig.suptitle(f"Epoch {epoch}", fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


# ── argument parsing ─────────────────────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root",   type=str,   default="fmri")
    p.add_argument("--out_dir",     type=str,   default=None)
    # data
    p.add_argument("--val_frac",    type=float, default=0.15)
    p.add_argument("--num_workers", type=int,   default=4)
    p.add_argument("--seed",        type=int,   default=42)
    # training
    p.add_argument("--epochs",      type=int,   default=200)
    p.add_argument("--batch_size",  type=int,   default=2)
    p.add_argument("--lr",          type=float, default=3e-4)
    p.add_argument("--amp",    dest="amp", action="store_true")
    p.add_argument("--no-amp", dest="amp", action="store_false")
    p.set_defaults(amp=True)
    # architecture
    p.add_argument("--diff_base",   type=int,   default=64)
    p.add_argument("--t_dim",       type=int,   default=256)
    p.add_argument("--n_levels",    type=int,   default=2)
    p.add_argument("--T",           type=int,   default=1000)
    p.add_argument("--ddim_steps",    type=int,   default=50,
                   help="DDIM steps used during validation sampling")
    p.add_argument("--cfg_drop_prob", type=float, default=0.15,
                   help="Fraction of training samples where conditioning is dropped (CFG)")
    p.add_argument("--guidance_scale", type=float, default=3.0,
                   help="CFG guidance scale used during validation sampling (1.0 = no guidance)")
    p.add_argument("--vis_every",     type=int,   default=5,
                   help="Save a prediction PNG every N epochs (0 to disable)")
    return p.parse_args()


# ── data ─────────────────────────────────────────────────────────────────────

def make_loaders(args):
    return build_loaders(args, augment=False)


# ── training ─────────────────────────────────────────────────────────────────

def train(args, device):
    out_dir = Path(args.out_dir or "runs/paired_diff")
    out_dir.mkdir(parents=True, exist_ok=True)

    train_dl, val_dl = make_loaders(args)

    model = build_paired_diffusion(
        base=args.diff_base,
        t_dim=args.t_dim,
        n_levels=args.n_levels,
        T=args.T,
    ).to(device)

    opt    = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    print(f"[PairedDiffusion] params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Train {len(train_dl.dataset)} / Val {len(val_dl.dataset)} samples")

    # fixed sample used for periodic visualisation
    vis_x, vis_y = next(iter(val_dl))
    vis_x, vis_y = vis_x.to(device), vis_y.to(device)

    best_val = float("inf")

    for epoch in range(1, args.epochs + 1):
        # ── train ──
        model.train()
        tr_loss = 0.0
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            with torch.cuda.amp.autocast(enabled=args.amp):
                loss = model.p_loss(x_post=y, x_pre=x, cfg_drop_prob=args.cfg_drop_prob)

            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)

            tr_loss += loss.item()

        tr_loss /= len(train_dl)

        # ── validate ──
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_dl:
                x, y = x.to(device), y.to(device)
                with torch.cuda.amp.autocast(enabled=args.amp):
                    val_loss += model.p_loss(x_post=y, x_pre=x).item()
        val_loss /= len(val_dl)

        print(
            f"Epoch {epoch:4d}/{args.epochs}  "
            f"train_loss={tr_loss:.4f}  val_loss={val_loss:.4f}"
        )

        ckpt = {
            "denoiser": model.denoiser.state_dict(),
            "args": {
                "diff_base": args.diff_base,
                "t_dim":     args.t_dim,
                "n_levels":  args.n_levels,
                "T":         args.T,
            },
        }
        torch.save(ckpt, out_dir / "last.pt")
        if val_loss < best_val:
            best_val = val_loss
            torch.save(ckpt, out_dir / "best.pt")
            print(f"  → new best ({best_val:.4f})")

        if args.vis_every > 0 and epoch % args.vis_every == 0:
            with torch.no_grad():
                y_pred = model.generate(vis_x, ddim_steps=args.ddim_steps, guidance_scale=args.guidance_scale)
            save_path = out_dir / f"pred_epoch_{epoch:04d}.png"
            visualize_pred(vis_x[0], vis_y[0], y_pred[0], epoch, save_path)
            print(f"  → saved visualisation: {save_path}")

    print(f"Done. Best checkpoint: {out_dir / 'best.pt'}")


# ── entry point ──────────────────────────────────────────────────────────────

def main():
    args = get_args()
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    train(args, device)


if __name__ == "__main__":
    main()
