"""
PairedLatentDiffusion (3D LDM) training — two-stage.

Stage 1: train the KL-autoencoder to reconstruct fMRI volumes.
Stage 2: freeze the AE; train the latent diffusion U-Net.

Usage:
    # Stage 1
    python scripts/train_ldm3d.py --stage 1 --out_dir runs/ldm3d

    # Stage 2 (requires a Stage-1 checkpoint)
    python scripts/train_ldm3d.py --stage 2 --ae_ckpt runs/ldm3d/stage1_best.pt \\
        --out_dir runs/ldm3d
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split

sys.path.insert(0, str(Path(__file__).parent.parent))

from moyamoya.dataset import PrePostFMRI
from moyamoya.transform import ToChannelsFirstAndNormalize
from moyamoya.utils import seed_everything
from moyamoya.models.ldm3d import build_paired_latent_diffusion, build_autoencoder


# ── helpers ───────────────────────────────────────────────────────────────────

def _to_np(vol: torch.Tensor) -> np.ndarray:
    v = vol.squeeze().cpu().float().numpy()
    lo, hi = np.percentile(v[v != 0], [1, 99]) if (v != 0).any() else (v.min(), v.max())
    return np.clip((v - lo) / (hi - lo + 1e-8), 0, 1)


def _mid(arr: np.ndarray, axis: int) -> int:
    return arr.shape[axis] // 2


def visualize_pred(x_pre, x_post_real, x_post_pred, epoch: int, save_path: Path):
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

    fig.suptitle(f"LDM3D Epoch {epoch}", fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


# ── args ─────────────────────────────────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--stage",       type=int,   default=1,  choices=[1, 2],
                   help="1 = train AE, 2 = train latent diffusion")
    p.add_argument("--ae_ckpt",     type=str,   default=None,
                   help="Stage-1 checkpoint path (required for stage 2)")
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
    # autoencoder architecture
    p.add_argument("--z_channels",  type=int,   default=4)
    p.add_argument("--embed_dim",   type=int,   default=4)
    p.add_argument("--ae_ch",       type=int,   default=64)
    p.add_argument("--ae_res_blocks", type=int, default=2)
    p.add_argument("--ae_resolution", type=int, default=64)
    p.add_argument("--kl_weight",   type=float, default=1e-6)
    # latent U-Net architecture
    p.add_argument("--diff_base",   type=int,   default=64)
    p.add_argument("--t_dim",       type=int,   default=256)
    p.add_argument("--n_levels",    type=int,   default=3)
    p.add_argument("--T",           type=int,   default=1000)
    p.add_argument("--ddim_steps",     type=int,   default=50)
    p.add_argument("--cfg_drop_prob",  type=float, default=0.15,
                   help="Fraction of training samples where conditioning is dropped (CFG)")
    p.add_argument("--guidance_scale", type=float, default=3.0,
                   help="CFG guidance scale used during validation sampling (1.0 = no guidance)")
    p.add_argument("--vis_every",      type=int,   default=5)
    return p.parse_args()


# ── data ─────────────────────────────────────────────────────────────────────

def make_loaders(args):
    tfm = ToChannelsFirstAndNormalize(nonzero_mask=True)
    ds  = PrePostFMRI(root_dir=args.data_root, transform=tfm, strict=False)
    n_val   = max(1, int(len(ds) * args.val_frac))
    n_train = len(ds) - n_val
    g = torch.Generator().manual_seed(args.seed)
    train_ds, val_ds = random_split(ds, [n_train, n_val], generator=g)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          num_workers=args.num_workers, pin_memory=True)
    val_dl   = DataLoader(val_ds,   batch_size=1, shuffle=False,
                          num_workers=args.num_workers, pin_memory=True)
    return train_dl, val_dl


# ── stage 1: autoencoder ─────────────────────────────────────────────────────

def train_stage1(args, device):
    out_dir = Path(args.out_dir or "runs/ldm3d")
    out_dir.mkdir(parents=True, exist_ok=True)

    train_dl, val_dl = make_loaders(args)

    ae = build_autoencoder(
        z_channels=args.z_channels,
        embed_dim=args.embed_dim,
        ch=args.ae_ch,
        num_res_blocks=args.ae_res_blocks,
        resolution=args.ae_resolution,
        kl_weight=args.kl_weight,
    ).to(device)

    opt    = torch.optim.AdamW(ae.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    print(f"[Stage 1 – AE] params: {sum(p.numel() for p in ae.parameters()):,}")
    print(f"  Train {len(train_dl.dataset)} / Val {len(val_dl.dataset)} samples")

    best_val = float("inf")

    for epoch in range(1, args.epochs + 1):
        ae.train()
        tr_loss = 0.0
        for x, y in train_dl:
            # train on both pre and post volumes independently
            for vol in (x.to(device), y.to(device)):
                with torch.cuda.amp.autocast(enabled=args.amp):
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
                for vol in (x.to(device), y.to(device)):
                    with torch.cuda.amp.autocast(enabled=args.amp):
                        val_loss += ae.rec_loss(vol).item()
        val_loss /= 2 * len(val_dl)

        print(f"Epoch {epoch:4d}/{args.epochs}  "
              f"train_loss={tr_loss:.4f}  val_loss={val_loss:.4f}")

        ckpt = {
            "ae": ae.state_dict(),
            "args": vars(args),
        }
        torch.save(ckpt, out_dir / "stage1_last.pt")
        if val_loss < best_val:
            best_val = val_loss
            torch.save(ckpt, out_dir / "stage1_best.pt")
            print(f"  → new best AE ({best_val:.4f})")

    print(f"Stage 1 done. Best checkpoint: {out_dir / 'stage1_best.pt'}")
    return out_dir / "stage1_best.pt"


# ── stage 2: latent diffusion ─────────────────────────────────────────────────

def train_stage2(args, device):
    if args.ae_ckpt is None:
        raise ValueError("--ae_ckpt is required for stage 2")

    out_dir = Path(args.out_dir or "runs/ldm3d")
    out_dir.mkdir(parents=True, exist_ok=True)

    train_dl, val_dl = make_loaders(args)

    model = build_paired_latent_diffusion(
        z_channels=args.z_channels,
        embed_dim=args.embed_dim,
        ae_ch=args.ae_ch,
        ae_res_blocks=args.ae_res_blocks,
        ae_resolution=args.ae_resolution,
        kl_weight=args.kl_weight,
        diff_base=args.diff_base,
        t_dim=args.t_dim,
        n_levels=args.n_levels,
        T=args.T,
    ).to(device)

    # load AE weights and freeze
    ae_state = torch.load(args.ae_ckpt, map_location=device)["ae"]
    model.ae.load_state_dict(ae_state)
    for p in model.ae.parameters():
        p.requires_grad_(False)
    model.ae.eval()

    opt    = torch.optim.AdamW(model.denoiser.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    n_denoiser = sum(p.numel() for p in model.denoiser.parameters())
    print(f"[Stage 2 – Latent Diffusion] denoiser params: {n_denoiser:,}")
    print(f"  Train {len(train_dl.dataset)} / Val {len(val_dl.dataset)} samples")

    vis_x, vis_y = next(iter(val_dl))
    vis_x, vis_y = vis_x.to(device), vis_y.to(device)

    best_val = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.denoiser.train()
        tr_loss = 0.0
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            with torch.cuda.amp.autocast(enabled=args.amp):
                loss = model.p_loss(x_post=y, x_pre=x, cfg_drop_prob=args.cfg_drop_prob)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.denoiser.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            tr_loss += loss.item()
        tr_loss /= len(train_dl)

        model.denoiser.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_dl:
                x, y = x.to(device), y.to(device)
                with torch.cuda.amp.autocast(enabled=args.amp):
                    val_loss += model.p_loss(x_post=y, x_pre=x).item()
        val_loss /= len(val_dl)

        print(f"Epoch {epoch:4d}/{args.epochs}  "
              f"train_loss={tr_loss:.4f}  val_loss={val_loss:.4f}")

        ckpt = {
            "ae":       model.ae.state_dict(),
            "denoiser": model.denoiser.state_dict(),
            "args":     vars(args),
        }
        torch.save(ckpt, out_dir / "stage2_last.pt")
        if val_loss < best_val:
            best_val = val_loss
            torch.save(ckpt, out_dir / "stage2_best.pt")
            print(f"  → new best ({best_val:.4f})")

        if args.vis_every > 0 and epoch % args.vis_every == 0:
            with torch.no_grad():
                y_pred = model.generate(vis_x, ddim_steps=args.ddim_steps, guidance_scale=args.guidance_scale)
            save_path = out_dir / f"pred_epoch_{epoch:04d}.png"
            visualize_pred(vis_x[0], vis_y[0], y_pred[0], epoch, save_path)
            print(f"  → saved visualisation: {save_path}")

    print(f"Stage 2 done. Best checkpoint: {out_dir / 'stage2_best.pt'}")


# ── entry point ───────────────────────────────────────────────────────────────

def main():
    args = get_args()
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if args.stage == 1:
        train_stage1(args, device)
    else:
        train_stage2(args, device)


if __name__ == "__main__":
    main()
