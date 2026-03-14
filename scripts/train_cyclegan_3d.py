"""
train_fmri_3d_cyclegan.py
─────────────────────────
Train a 3D CycleGAN on unpaired pre/post fMRI volumes using
3D-CycleGan-Pytorch-MedImaging.

Data layout expected:
    <dataroot>/pre_surgery/*.nii.gz          (domain A — 200 volumes)
    <dataroot>/6_months_post_surgery/*.nii.gz (domain B — 200 volumes)

Usage:
    python train_fmri_3d_cyclegan.py --dataroot fmri/ --name fmri_3d_cyclegan

Key flags:
    --dataroot          Root directory containing pre_surgery/ and post_surgery/
    --name              Experiment name (checkpoints and HTML go here)
    --ngf               Generator base filters (default 32; use 16 if OOM)
    --n_epochs          Epochs at fixed LR
    --n_epochs_decay    Epochs with linear LR decay
    --display_freq      Save visualization every N iterations
    --norm_mode         zscore (default) | minmax
"""

from __future__ import annotations

import sys
import os
import time
import random
import argparse
from pathlib import Path
from collections import defaultdict
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import nibabel as nib
import numpy as np

# ── 3D CycleGAN repo ─────────────────────────────────────────────────────────
HERE    = Path(__file__).resolve().parent
_ROOT   = HERE.parent
REPO_3D = _ROOT / "third_party" / "3D-CycleGan-Pytorch-MedImaging"
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(REPO_3D))

from models.cycle_gan_model import CycleGANModel   # noqa: E402
from models import networks3D                        # noqa: E402
from moyamoya.viz.html_viz import VolumeVisualizer   # noqa: E402


# ─────────────────────────────────────────────────────────────────────────────
# Masked CycleGAN — cycle & identity losses computed only over brain voxels
# ─────────────────────────────────────────────────────────────────────────────

class MaskedCycleGANModel(CycleGANModel):
    """
    Subclass of CycleGANModel that accepts per-volume brain masks and uses
    them to compute masked L1 losses for cycle consistency and identity.

    The GAN losses are unchanged (discriminator sees full volumes).
    """

    def set_input(self, input):
        AtoB = self.opt.which_direction == 'AtoB'
        self.real_A = input[0 if AtoB else 1].to(self.device)
        self.real_B = input[1 if AtoB else 0].to(self.device)
        self.mask_A = input[2].to(self.device)   # (B,1,D,H,W) brain mask for A
        self.mask_B = input[3].to(self.device)   # (B,1,D,H,W) brain mask for B

    @staticmethod
    def _masked_l1(pred: torch.Tensor, target: torch.Tensor,
                   mask: torch.Tensor) -> torch.Tensor:
        """Mean L1 over masked (brain) voxels only."""
        diff  = (pred - target).abs() * mask
        denom = mask.sum().clamp_min(1.0)
        return diff.sum() / denom

    def backward_G(self):
        lambda_idt = self.opt.lambda_identity
        lambda_A   = self.opt.lambda_A
        lambda_B   = self.opt.lambda_B

        # Identity loss (masked)
        if lambda_idt > 0:
            self.idt_A      = self.netG_A(self.real_B)
            self.loss_idt_A = self._masked_l1(self.idt_A, self.real_B,
                                              self.mask_B) * lambda_B * lambda_idt
            self.idt_B      = self.netG_B(self.real_A)
            self.loss_idt_B = self._masked_l1(self.idt_B, self.real_A,
                                              self.mask_A) * lambda_A * lambda_idt
        else:
            self.loss_idt_A = 0
            self.loss_idt_B = 0

        # GAN losses (unchanged — full volume)
        self.loss_G_A = self.criterionGAN(self.netD_A(self.fake_B), True)
        self.loss_G_B = self.criterionGAN(self.netD_B(self.fake_A), True)

        # Cycle losses (masked)
        self.loss_cycle_A = self._masked_l1(self.rec_A, self.real_A,
                                            self.mask_A) * lambda_A
        self.loss_cycle_B = self._masked_l1(self.rec_B, self.real_B,
                                            self.mask_B) * lambda_B

        self.loss_G = (self.loss_G_A + self.loss_G_B
                       + self.loss_cycle_A + self.loss_cycle_B
                       + self.loss_idt_A   + self.loss_idt_B)
        self.loss_G.backward()


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class FMRIDataset3D(Dataset):
    """
    Unpaired 3D fMRI dataset.

    Loads all pre- and post-surgery NIfTI volumes into RAM, normalizes them,
    and pads to a size divisible by `pad_divisor`.

    A binary brain mask is computed per volume (raw > mask_thresh * max).
    Background voxels are set to -1 in the input, and the mask is returned
    so the cycle/identity losses are computed only over brain voxels.

    For each index i, returns:
        A, mask_A = pre_surgery[i % n_A]
        B, mask_B = post_surgery[random index]   ← unpaired CycleGAN assumption
    """

    def __init__(
        self,
        root_dir: str | Path,
        pre_dirname: str = "pre_surgery",
        post_dirname: str = "6_months_post_surgery",
        norm_mode: str = "zscore",      # zscore | minmax
        pad_divisor: int = 8,
        mask_thresh: float = 0.05,
    ):
        self.root_dir    = Path(root_dir)
        self.norm_mode   = norm_mode
        self.pad_divisor = pad_divisor
        self.mask_thresh = mask_thresh

        pre_dir  = self.root_dir / pre_dirname
        post_dir = self.root_dir / post_dirname

        if not pre_dir.is_dir():
            raise FileNotFoundError(f"Pre-surgery directory not found: {pre_dir}")
        if not post_dir.is_dir():
            raise FileNotFoundError(f"Post-surgery directory not found: {post_dir}")

        print(f"[FMRIDataset3D] Loading pre-surgery volumes from {pre_dir} ...")
        self.vols_A, self.masks_A = self._load_all(pre_dir)
        print(f"[FMRIDataset3D]   {len(self.vols_A)} volumes loaded (domain A)")

        print(f"[FMRIDataset3D] Loading post-surgery volumes from {post_dir} ...")
        self.vols_B, self.masks_B = self._load_all(post_dir)
        print(f"[FMRIDataset3D]   {len(self.vols_B)} volumes loaded (domain B)")

        if not self.vols_A or not self.vols_B:
            raise RuntimeError("No volumes found — check your dataroot and subdirectory names.")

        shape_A = self.vols_A[0].shape
        n_brain = self.masks_A[0].sum().item()
        n_total = self.masks_A[0].numel()
        print(f"[FMRIDataset3D] Volume shape after pad: {shape_A}  norm_mode={norm_mode}")
        print(f"[FMRIDataset3D] Brain mask coverage (first A vol): "
              f"{n_brain}/{n_total} = {100*n_brain/n_total:.1f}%")

    # ── loading ───────────────────────────────────────────────────────────

    def _load_all(self, vol_dir: Path):
        vols, masks = [], []
        for p in sorted(vol_dir.glob("*.nii.gz")):
            raw  = self._load_raw(p)
            mask = self._compute_mask(raw)
            vol  = self._normalize(raw)
            vol  = self._pad(vol)
            mask = self._pad_mask(mask)
            # Zero out background in the input so the generator ignores it
            vol  = vol * mask + (-1.0) * (1.0 - mask)
            vols.append(vol)
            masks.append(mask)
        return vols, masks

    @staticmethod
    def _load_raw(path: Path) -> torch.Tensor:
        img = nib.load(str(path))
        return torch.from_numpy(img.get_fdata(dtype="float32"))

    def _compute_mask(self, raw: torch.Tensor) -> torch.Tensor:
        """Threshold brain mask: 1 inside brain, 0 outside."""
        return (raw > raw.max() * self.mask_thresh).float()

    def _normalize(self, vol: torch.Tensor) -> torch.Tensor:
        if self.norm_mode == "zscore":
            mask = vol > vol.max() * 0.01
            if mask.any():
                vals = vol[mask]
                vol  = (vol - vals.mean()) / vals.std(unbiased=False).clamp_min(1e-6)
            return vol.clamp(-3.0, 3.0) / 3.0
        elif self.norm_mode == "minmax":
            lo, hi = vol.min(), vol.max()
            if hi > lo:
                return (vol - lo) / (hi - lo) * 2.0 - 1.0
            return torch.zeros_like(vol)
        else:
            raise ValueError(f"Unknown norm_mode: {self.norm_mode}")

    def _pad(self, vol: torch.Tensor) -> torch.Tensor:
        """Pad spatial dims to next multiple of pad_divisor."""
        d = self.pad_divisor
        D, H, W = vol.shape
        pD = (d - D % d) % d
        pH = (d - H % d) % d
        pW = (d - W % d) % d
        if pD or pH or pW:
            # F.pad takes (W_lo, W_hi, H_lo, H_hi, D_lo, D_hi) for 3-D
            vol = F.pad(vol, (0, pW, 0, pH, 0, pD), mode="constant", value=-1.0)
        return vol

    def _pad_mask(self, mask: torch.Tensor) -> torch.Tensor:
        """Pad mask with 0 (background) to match volume padding."""
        d = self.pad_divisor
        D, H, W = mask.shape
        pD = (d - D % d) % d
        pH = (d - H % d) % d
        pW = (d - W % d) % d
        if pD or pH or pW:
            mask = F.pad(mask, (0, pW, 0, pH, 0, pD), mode="constant", value=0.0)
        return mask

    # ── dataset interface ─────────────────────────────────────────────────

    def __len__(self) -> int:
        return max(len(self.vols_A), len(self.vols_B))

    def __getitem__(self, idx: int):
        a_idx = idx % len(self.vols_A)
        b_idx = random.randint(0, len(self.vols_B) - 1)
        A      = self.vols_A[a_idx]
        mask_A = self.masks_A[a_idx]
        B      = self.vols_B[b_idx]
        mask_B = self.masks_B[b_idx]
        # Add channel dim → (1, D, H, W)
        return A.unsqueeze(0), B.unsqueeze(0), mask_A.unsqueeze(0), mask_B.unsqueeze(0)


# ─────────────────────────────────────────────────────────────────────────────
# Options
# ─────────────────────────────────────────────────────────────────────────────

def build_opt(args: argparse.Namespace) -> argparse.Namespace:
    """Construct the flat opt namespace that CycleGANModel expects."""
    opt = argparse.Namespace()

    # ── paths ────────────────────────────────────────────────────────────
    opt.checkpoints_dir = args.checkpoints_dir
    opt.name            = args.name

    # ── hardware ─────────────────────────────────────────────────────────
    if torch.cuda.is_available() and args.gpu_ids:
        opt.gpu_ids = [int(x) for x in args.gpu_ids.split(",")]
    else:
        opt.gpu_ids = []

    # ── model architecture ───────────────────────────────────────────────
    opt.input_nc   = 1
    opt.output_nc  = 1
    opt.ngf        = args.ngf
    opt.ndf        = args.ndf
    opt.netG       = args.netG
    opt.netD       = "n_layers"
    opt.n_layers_D = 3
    opt.norm       = "instance"
    opt.no_dropout = True          # CycleGAN default
    opt.init_type  = "normal"
    opt.init_gain  = 0.02

    # ── losses ───────────────────────────────────────────────────────────
    opt.lambda_A        = args.lambda_A
    opt.lambda_B        = args.lambda_B
    opt.lambda_identity = args.lambda_identity
    opt.lambda_co_A     = 0.0     # correlation coeff loss disabled
    opt.lambda_co_B     = 0.0
    opt.no_lsgan        = False   # use LSGAN (MSE)
    opt.pool_size       = args.pool_size

    # ── training schedule ────────────────────────────────────────────────
    opt.beta1          = 0.5
    opt.lr             = args.lr
    opt.lr_policy      = "lambda"
    opt.epoch_count    = 1
    opt.niter          = args.n_epochs
    opt.niter_decay    = args.n_epochs_decay
    opt.lr_decay_iters = 50

    # ── misc ─────────────────────────────────────────────────────────────
    opt.isTrain        = True
    opt.continue_train = args.continue_train
    opt.which_epoch    = "latest"
    opt.which_direction = "AtoB"
    opt.verbose        = False
    opt.model          = "cycle_gan"

    return opt


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Train a 3D CycleGAN on unpaired pre/post fMRI volumes",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # ── data ────────────────────────────────────────────────────────────
    p.add_argument("--dataroot",       default="fmri",
                   help="Root directory containing pre_surgery/ and 6_months_post_surgery/")
    p.add_argument("--pre_dirname",    default="pre_surgery")
    p.add_argument("--post_dirname",   default="6_months_post_surgery")
    p.add_argument("--norm_mode",      default="zscore", choices=["zscore", "minmax"],
                   help="Per-volume normalization: zscore or minmax to [-1,1]")
    p.add_argument("--mask_thresh",    type=float, default=0.05,
                   help="Brain mask threshold as fraction of per-volume max intensity")

    # ── experiment ──────────────────────────────────────────────────────
    p.add_argument("--name",            default="fmri_3d_cyclegan")
    p.add_argument("--checkpoints_dir", default="checkpoints")
    p.add_argument("--continue_train",  action="store_true")

    # ── hardware ────────────────────────────────────────────────────────
    p.add_argument("--gpu_ids",     default="0",
                   help="Comma-separated GPU ids, e.g. '0,1'. Empty string for CPU.")
    p.add_argument("--batch_size",  type=int, default=1)
    p.add_argument("--num_workers", type=int, default=4)

    # ── network ─────────────────────────────────────────────────────────
    p.add_argument("--ngf",   type=int, default=48,
                   help="Generator base filters. Use 32 or 16 if OOM.")
    p.add_argument("--ndf",   type=int, default=32,
                   help="Discriminator base filters.")
    p.add_argument("--netG",  default="resnet_9blocks",
                   choices=["resnet_9blocks", "resnet_6blocks"],
                   help="Generator architecture.")

    # ── losses ──────────────────────────────────────────────────────────
    p.add_argument("--lambda_A",        type=float, default=15.0)
    p.add_argument("--lambda_B",        type=float, default=15.0)
    p.add_argument("--lambda_identity", type=float, default=0.1)
    p.add_argument("--pool_size",       type=int,   default=50)

    # ── training schedule ───────────────────────────────────────────────
    p.add_argument("--lr",             type=float, default=2e-4)
    p.add_argument("--n_epochs",       type=int,   default=100,
                   help="Epochs at the initial learning rate")
    p.add_argument("--n_epochs_decay", type=int,   default=100,
                   help="Epochs with linear LR decay to zero")

    # ── logging & visualization ──────────────────────────────────────────
    p.add_argument("--print_freq",    type=int, default=10,
                   help="Print losses every N iterations")
    p.add_argument("--display_freq",  type=int, default=10,
                   help="Save HTML visualization snapshot every N iterations")
    p.add_argument("--save_epoch_freq", type=int, default=10,
                   help="Save checkpoint every N epochs")
    p.add_argument("--viz_refresh",   type=int, default=30,
                   help="HTML auto-refresh interval in seconds")

    return p


# ─────────────────────────────────────────────────────────────────────────────
# Loss helpers
# ─────────────────────────────────────────────────────────────────────────────

class RunningAvg:
    def __init__(self):
        self.s, self.n = 0.0, 0

    def update(self, v):
        if v is None:
            return
        v = float(v.item()) if hasattr(v, "item") else float(v)
        if not (v != v or abs(v) == float("inf")):
            self.s += v
            self.n += 1

    @property
    def mean(self):
        return self.s / max(1, self.n)


# ─────────────────────────────────────────────────────────────────────────────
# Paired visualization batch
# ─────────────────────────────────────────────────────────────────────────────

def load_viz_pair(
    root_dir: str | Path,
    pre_dirname: str,
    post_dirname: str,
    norm_mode: str,
    pad_divisor: int = 8,
    mask_thresh: float = 0.05,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Find the first subject that has matching filenames in both pre and post
    directories, load and normalize both volumes, and return them as a fixed
    (A, B, mask_A, mask_B) visualization pair.

    The pair is the SAME SUBJECT so real_A vs fake_B is a meaningful comparison.
    """
    pre_dir  = Path(root_dir) / pre_dirname
    post_dir = Path(root_dir) / post_dirname

    pre_files  = {p.name: p for p in sorted(pre_dir.glob("*.nii.gz"))}
    post_files = {p.name: p for p in sorted(post_dir.glob("*.nii.gz"))}
    shared     = sorted(set(pre_files) & set(post_files))

    if not shared:
        print("[WARN] No filename-matched subjects found for paired visualization; "
              "falling back to first pre and first post volume (different subjects).")
        pre_path  = sorted(pre_dir.glob("*.nii.gz"))[0]
        post_path = sorted(post_dir.glob("*.nii.gz"))[0]
    else:
        name      = shared[0]
        pre_path  = pre_files[name]
        post_path = post_files[name]
        print(f"[INFO] Paired viz subject: {name}")

    dummy_ds = FMRIDataset3D.__new__(FMRIDataset3D)
    dummy_ds.norm_mode   = norm_mode
    dummy_ds.pad_divisor = pad_divisor
    dummy_ds.mask_thresh = mask_thresh

    raw_A  = dummy_ds._load_raw(pre_path)
    raw_B  = dummy_ds._load_raw(post_path)
    mask_A = dummy_ds._pad_mask(dummy_ds._compute_mask(raw_A))
    mask_B = dummy_ds._pad_mask(dummy_ds._compute_mask(raw_B))
    A = dummy_ds._pad(dummy_ds._normalize(raw_A))
    B = dummy_ds._pad(dummy_ds._normalize(raw_B))
    A = (A * mask_A + (-1.0) * (1.0 - mask_A)).unsqueeze(0).unsqueeze(0)  # (1,1,D,H,W)
    B = (B * mask_B + (-1.0) * (1.0 - mask_B)).unsqueeze(0).unsqueeze(0)
    mask_A = mask_A.unsqueeze(0).unsqueeze(0)
    mask_B = mask_B.unsqueeze(0).unsqueeze(0)

    return A, B, mask_A, mask_B


@torch.no_grad()
def run_viz_forward(model, viz_A: torch.Tensor, viz_B: torch.Tensor,
                    mask_A: torch.Tensor, mask_B: torch.Tensor) -> dict:
    """Run a full CycleGAN forward pass on the fixed viz pair and return visuals."""
    device = model.device
    model.set_input((viz_A.to(device), viz_B.to(device),
                     mask_A.to(device), mask_B.to(device)))
    model.forward()
    return model.get_current_visuals()


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = build_parser()
    args   = parser.parse_args()

    # ── checkpoints dir ──────────────────────────────────────────────────
    ckpt_dir = Path(args.checkpoints_dir) / args.name
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ── HTML visualizer ──────────────────────────────────────────────────
    web_dir = ckpt_dir / "web"
    viz = VolumeVisualizer(
        web_dir      = web_dir,
        title        = f"3D CycleGAN — {args.name}",
        refresh_secs = args.viz_refresh,
    )
    print(f"[INFO] HTML log: {web_dir / 'index.html'}")

    # ── fixed paired visualization batch ─────────────────────────────────
    # Same subject for A and B so fake_B vs real_B is a true comparison.
    viz_A, viz_B, viz_mask_A, viz_mask_B = load_viz_pair(
        root_dir     = args.dataroot,
        pre_dirname  = args.pre_dirname,
        post_dirname = args.post_dirname,
        norm_mode    = args.norm_mode,
        mask_thresh  = args.mask_thresh,
    )

    # ── dataset & loader ─────────────────────────────────────────────────
    dataset = FMRIDataset3D(
        root_dir     = args.dataroot,
        pre_dirname  = args.pre_dirname,
        post_dirname = args.post_dirname,
        norm_mode    = args.norm_mode,
        mask_thresh  = args.mask_thresh,
    )
    loader = DataLoader(
        dataset,
        batch_size  = args.batch_size,
        shuffle     = True,
        num_workers = args.num_workers,
        pin_memory  = torch.cuda.is_available(),
        drop_last   = True,
    )
    print(f"[INFO] Dataset: {len(dataset)} samples per epoch, {len(loader)} iterations")

    # ── model ────────────────────────────────────────────────────────────
    opt   = build_opt(args)
    model = MaskedCycleGANModel()
    model.initialize(opt)
    model.setup(opt)

    device = model.device
    print(f"[INFO] Using device: {device}")

    # ── loss log ─────────────────────────────────────────────────────────
    loss_log_path = ckpt_dir / "loss_log.txt"
    loss_log = loss_log_path.open("a", buffering=1)
    loss_log.write(f"\n{'='*60}\n Training started — {args.name}\n{'='*60}\n")

    # ── training loop ─────────────────────────────────────────────────────
    total_iters  = 0
    start_time   = time.time()
    max_epoch    = opt.niter + opt.niter_decay

    for epoch in range(opt.epoch_count, max_epoch + 1):
        epoch_start  = time.time()
        loss_meters  = defaultdict(RunningAvg)

        for i, (A, B, mask_A, mask_B) in enumerate(loader):
            total_iters += args.batch_size

            model.set_input((A, B, mask_A, mask_B))
            model.optimize_parameters()

            # ── record losses ─────────────────────────────────────────
            losses = model.get_current_losses()
            for k, v in losses.items():
                loss_meters[k].update(v)

            # ── print ─────────────────────────────────────────────────
            if total_iters % args.print_freq == 0:
                loss_str = "  ".join(f"{k}: {float(v):.4f}" for k, v in losses.items())
                elapsed  = (time.time() - start_time) / 60
                msg = f"[epoch {epoch}/{max_epoch}]  [iter {i+1}/{len(loader)}]  " \
                      f"{loss_str}  | {elapsed:.1f} min"
                print(msg)
                loss_log.write(msg + "\n")

            # ── visualization (fixed paired subject) ──────────────────
            if total_iters % args.display_freq == 0:
                visuals = run_viz_forward(model, viz_A, viz_B, viz_mask_A, viz_mask_B)
                viz.save_volumes(epoch=epoch, iters=total_iters, volumes=visuals)

        # ── epoch summary ─────────────────────────────────────────────
        mean_losses = {k: m.mean for k, m in loss_meters.items()}
        loss_summary = "  ".join(f"{k}: {v:.4f}" for k, v in sorted(mean_losses.items()))
        epoch_time   = (time.time() - epoch_start) / 60
        summary = f"\n[EPOCH {epoch}/{max_epoch}]  epoch_time: {epoch_time:.1f} min  " \
                  f"mean losses — {loss_summary}\n"
        print(summary)
        loss_log.write(summary)

        # ── checkpoint ────────────────────────────────────────────────
        model.save_networks("latest")
        if epoch % args.save_epoch_freq == 0:
            model.save_networks(epoch)
            print(f"[INFO] Saved checkpoint for epoch {epoch}")

        # ── LR schedule ───────────────────────────────────────────────
        model.update_learning_rate()

    loss_log.write("[DONE] Training complete.\n")
    loss_log.close()
    print("[DONE] Training complete.")


if __name__ == "__main__":
    main()
