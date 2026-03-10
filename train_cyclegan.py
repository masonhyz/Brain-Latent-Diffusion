"""
minimal_train_cyclegan_prepost.py

Uses the CycleGAN repo's TrainOptions + CycleGANModel, but uses YOUR
DataLoader built from:

  base_ds = PrePostFMRI(root_dir="...", strict=True, return_paths=True)
  cyclegan_ds = CycleGANDictWrapper(base_ds, direction="AtoB")

No paired supervision is added (standard CycleGAN losses only).

Run from a context where `pytorch_CycleGAN_and_pix2pix` is importable.
"""

from __future__ import annotations

import os
import time
import random
from pathlib import Path
from typing import Dict, Any

import torch
from torch.utils.data import DataLoader

from pytorch_CycleGAN_and_pix2pix.options.train_options import TrainOptions
from pytorch_CycleGAN_and_pix2pix.models.cycle_gan_model import CycleGANModel

from dataset import PrePostFMRI
from cyclegan_dataset_wrapper import CycleGANDictWrapper


import math
from collections import defaultdict

class RunningAvg:
    def __init__(self):
        self.sum = 0.0
        self.n = 0

    def update(self, v: float, k: int = 1):
        if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
            return
        self.sum += float(v) * k
        self.n += int(k)

    @property
    def mean(self) -> float:
        return self.sum / max(1, self.n)

def as_float(x):
    # repo losses can be python floats or 0-d tensors
    if hasattr(x, "item"):
        return float(x.item())
    return float(x)

@torch.no_grad()
def discriminator_rates(model, gan_mode: str):
    """
    Returns proxy stats:
      D_A_real_rate, D_A_fake_rate, D_B_real_rate, D_B_fake_rate
    Interpretation (rough):
      - "real_rate": fraction of D outputs classified as real on real samples
      - "fake_rate": fraction of D outputs classified as fake on fake samples

    For LSGAN: target real=1, fake=0. We threshold at 0.5.
    For vanilla: logits; threshold at 0.
    For wgangp: not meaningful (returns None).
    """
    if gan_mode == "wgangp":
        return None

    thr = 0.5 if gan_mode == "lsgan" else 0.0

    # D_A judges domain B (real_B vs fake_B)
    pred_real_B = model.netD_A(model.real_B)
    pred_fake_B = model.netD_A(model.fake_B.detach())

    # D_B judges domain A (real_A vs fake_A)
    pred_real_A = model.netD_B(model.real_A)
    pred_fake_A = model.netD_B(model.fake_A.detach())

    def rate_real(pred):
        return float((pred > thr).float().mean().item())

    def rate_fake(pred):
        return float((pred <= thr).float().mean().item())

    return {
        "D_A_real_rate": rate_real(pred_real_B),
        "D_A_fake_rate": rate_fake(pred_fake_B),
        "D_B_real_rate": rate_real(pred_real_A),
        "D_B_fake_rate": rate_fake(pred_fake_A),
    }


def set_seed(seed: int = 0):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dirs(opt):
    ckpt_dir = Path(opt.checkpoints_dir) / opt.name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    return ckpt_dir


def main():
    # -------------------------
    # 1) Parse CycleGAN options
    # -------------------------
    opt = TrainOptions().parse()
    
    if not hasattr(opt, "device"):
        opt.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Also ensure gpu_ids exists (some code uses both)
    if not hasattr(opt, "gpu_ids"):
        opt.gpu_ids = [0] if torch.cuda.is_available() else []

    # ---- IMPORTANT overrides for your setup (edit as needed) ----
    # Your data isn't in trainA/trainB; it's in pre_surgery/ and 6_months_post_surgery/
    # We only need dataroot to exist because BaseOptions requires it and prints it.
    # It is not used to build our DataLoader (we do that ourselves).
    # If you passed --dataroot fmri, make sure it exists.
    if opt.dataroot is None:
        raise ValueError("TrainOptions requires --dataroot. Provide any existing path (e.g., the fmri root).")

    # If you're training 3D, set channels to 1 (common for MRI/fMRI mean volumes)
    # Also set your 3D nets (must exist in networks.define_G/define_D).
    # You can override via CLI too, e.g.:
    #   --input_nc 1 --output_nc 1 --netG resnet_6blocks_3d --netD basic_3d --ngf 32 --ndf 32
    # Here we just ensure sensible defaults if user forgot.
    if opt.input_nc == 3:
        opt.input_nc = 1
    if opt.output_nc == 3:
        opt.output_nc = 1

    # Reduce filters for 3D to avoid OOM
    if opt.ngf == 64:
        opt.ngf = 32
    if opt.ndf == 64:
        opt.ndf = 32

    # Typically keep batch_size=1 for CycleGAN
    if opt.batch_size != 1:
        print(f"[WARN] CycleGAN typically uses batch_size=1; you set {opt.batch_size}")

    # reproducibility
    set_seed(0)

    # checkpoint folder
    ckpt_dir = ensure_dirs(opt)

    # -------------------------
    # 2) Build your dataset/loader
    # -------------------------
    # IMPORTANT: Use a REAL path here (not the placeholder from earlier).
    # This should contain:
    #   root_dir/pre_surgery/*.nii.gz
    #   root_dir/6_months_post_surgery/*.nii.gz
    #
    # If you want to use CLI: replace this hardcode with argparse or environment variable.
    fmri_root_dir = opt.dataroot  # simplest: reuse --dataroot as your dataset root

    base_ds = PrePostFMRI(
        root_dir=fmri_root_dir,
        pre_dirname="pre_surgery",
        post_dirname="6_months_post_surgery",
        strict=True,
        return_paths=True,
        transform=None,
    )

    # cyclegan_ds = CycleGANDictWrapper(base_ds, direction=opt.direction)

    cyclegan_ds = CycleGANDictWrapper(
        base_ds,
        direction="AtoB",
        pad_to=(92, 112, 92),   # pads D and W by +1 for your 91,109,91 data
        pad_mode="constant",
        pad_value=0.0,
        strict_pad=True,
    )

    loader = DataLoader(
        cyclegan_ds,
        batch_size=opt.batch_size,
        shuffle=(not opt.serial_batches),
        num_workers=opt.num_threads,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )


    # -------------------------
    # 3) Init CycleGAN model
    # -------------------------
    # CycleGANModel expects opt to have gpu_ids
    if not hasattr(opt, "gpu_ids"):
        opt.gpu_ids = [0] if torch.cuda.is_available() else []
    # In some repo variants, BaseModel sets self.device using gpu_ids.
    # Ensure proper isTrain
    opt.isTrain = True

    model = CycleGANModel(opt)

    # Some forks require calling setup() to create schedulers / load networks.
    # If your BaseModel has setup(), call it.
    if hasattr(model, "setup"):
        model.setup(opt)

    device = model.device if hasattr(model, "device") else (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    print(f"[INFO] Using device: {device}")

    # -------------------------
    # 4) Training loop (CycleGAN-native)
    # -------------------------
    total_iters = 0
    start_time = time.time()

    # If your TrainOptions defines epoch_count, n_epochs, n_epochs_decay:
    max_epoch = opt.n_epochs + opt.n_epochs_decay
    epoch_count = getattr(opt, "epoch_count", 1)

    best_metric = float("inf")  # lower is better
    best_epoch = -1

    for epoch in range(epoch_count, max_epoch + 1):
        epoch_start = time.time()

        # Track epoch means
        loss_meters = defaultdict(RunningAvg)
        rate_meters = defaultdict(RunningAvg)

        for i, batch in enumerate(loader):
            total_iters += opt.batch_size

            model.set_input(batch)
            model.optimize_parameters()

            # ---------- per-iter: record losses ----------
            if hasattr(model, "get_current_losses"):
                losses = model.get_current_losses()  # dict: name->float
                for k, v in losses.items():
                    loss_meters[k].update(as_float(v))

            # ---------- optional: discriminator "rates" proxy ----------
            # this adds a tiny compute overhead
            try:
                rates = discriminator_rates(model, opt.gan_mode)
                if rates is not None:
                    for k, v in rates.items():
                        rate_meters[k].update(v)
            except Exception:
                # don't crash training for reporting
                pass

            # ---------- your existing print ----------
            if total_iters % opt.print_freq == 0:
                losses_now = model.get_current_losses() if hasattr(model, "get_current_losses") else {}
                loss_str = " ".join([f"{k}:{as_float(v):.4f}" for k, v in losses_now.items()])
                mins = (time.time() - start_time) / 60.0
                print(f"[epoch {epoch}/{max_epoch}] [iter {i}] [total {total_iters}] {loss_str} | {mins:.1f} min")

            # ---------- your existing latest save ----------
            if total_iters % opt.save_latest_freq == 0:
                print(f"[INFO] Saving latest at iters={total_iters}, epoch={epoch}")
                if hasattr(model, "save_networks"):
                    model.save_networks("latest")
                else:
                    torch.save(
                        {
                            "epoch": epoch,
                            "total_iters": total_iters,
                            "netG_A": model.netG_A.state_dict(),
                            "netG_B": model.netG_B.state_dict(),
                            "netD_A": model.netD_A.state_dict(),
                            "netD_B": model.netD_B.state_dict(),
                            "optG": model.optimizer_G.state_dict(),
                            "optD": model.optimizer_D.state_dict(),
                            "opt": vars(opt),
                        },
                        ckpt_dir / "latest.pt",
                    )

        # -------------------------
        # end of epoch: reporting
        # -------------------------
        epoch_losses = {k: m.mean for k, m in loss_meters.items()}
        epoch_rates = {k: m.mean for k, m in rate_meters.items()}

        # Define a "best" criterion (lower is better):
        # Prefer the full generator objective if available; else fall back to cycle losses
        if "G_A" in epoch_losses and "G_B" in epoch_losses:
            # CycleGAN's printed losses don't always include "G" total, so we build a proxy:
            g_total = 0.0
            for key in ["G_A", "G_B", "cycle_A", "cycle_B", "idt_A", "idt_B"]:
                if key in epoch_losses:
                    if key == "cycle_A" or key == "cycle_B" or key == "idt_A" or key == "idt_B":
                        g_total += epoch_losses[key] / 100  # divide by 100 to give more weight to generator losses and identity loss
                    else:
                        g_total += epoch_losses[key]
            metric_name = "G_total_proxy"
            metric_value = g_total
        else:
            metric_name = "cycle_total"
            metric_value = epoch_losses.get("cycle_A", 0.0) + epoch_losses.get("cycle_B", 0.0)

        # Pretty print epoch summary
        loss_report = " ".join([f"{k}:{v:.4f}" for k, v in sorted(epoch_losses.items())])
        print(f"\n[EPOCH {epoch}] mean losses: {loss_report}")
        if epoch_rates:
            rate_report = " ".join([f"{k}:{v:.3f}" for k, v in sorted(epoch_rates.items())])
            print(f"[EPOCH {epoch}] D rates (proxy): {rate_report}")
        print(f"[EPOCH {epoch}] selection metric ({metric_name}): {metric_value:.4f}")

        # -------------------------
        # save best checkpoint
        # -------------------------
        if metric_value < best_metric:
            best_metric = metric_value
            best_epoch = epoch
            print(f"[INFO] New BEST at epoch {epoch}: {metric_name}={best_metric:.4f} -> saving 'best'")

            if hasattr(model, "save_networks"):
                model.save_networks("best")  # will create best_netG_A.pth etc. in repo-style
            else:
                torch.save(
                    {
                        "epoch": epoch,
                        "total_iters": total_iters,
                        "best_metric_name": metric_name,
                        "best_metric_value": best_metric,
                        "netG_A": model.netG_A.state_dict(),
                        "netG_B": model.netG_B.state_dict(),
                        "netD_A": model.netD_A.state_dict(),
                        "netD_B": model.netD_B.state_dict(),
                        "optG": model.optimizer_G.state_dict(),
                        "optD": model.optimizer_D.state_dict(),
                        "opt": vars(opt),
                    },
                    ckpt_dir / "best.pt",
                )

        # -------------------------
        # end epoch saves (your existing)
        # -------------------------
        if epoch % opt.save_epoch_freq == 0:
            print(f"[INFO] Saving epoch checkpoint: {epoch}")
            if hasattr(model, "save_networks"):
                model.save_networks("latest")
                model.save_networks(epoch)
            else:
                torch.save(
                    {
                        "epoch": epoch,
                        "total_iters": total_iters,
                        "netG_A": model.netG_A.state_dict(),
                        "netG_B": model.netG_B.state_dict(),
                        "netD_A": model.netD_A.state_dict(),
                        "netD_B": model.netD_B.state_dict(),
                        "optG": model.optimizer_G.state_dict(),
                        "optD": model.optimizer_D.state_dict(),
                        "opt": vars(opt),
                    },
                    ckpt_dir / f"epoch_{epoch}.pt",
                )

        # lr schedule
        if hasattr(model, "update_learning_rate"):
            model.update_learning_rate()

        print(f"[INFO] End of epoch {epoch} | epoch time {(time.time() - epoch_start)/60.0:.1f} min | best_epoch={best_epoch} best={best_metric:.4f}\n")



    print("[DONE] Training complete.")


if __name__ == "__main__":
    main()