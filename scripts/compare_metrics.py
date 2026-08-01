"""
compare_metrics.py

Runs inference for 7TCDM-3D LDM, LDM-3D, and identity baseline on all
training samples and prints a comparison table of MAE, MSE, SSIM, PSNR.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from skimage.metrics import structural_similarity, peak_signal_noise_ratio

from moyamoya.dataset import PrePostFMRI
from moyamoya.transform import ToChannelsFirstAndNormalize
from moyamoya.models.ldm_7tcdm3d import build_paired_latent_diffusion_7tcdm
from moyamoya.models.ldm3d import build_paired_latent_diffusion
from moyamoya.utils import make_union_mask

# ── config ────────────────────────────────────────────────────────────────────
TCDM_CKPT = "runs/ldm_7tcdm3d/stage2_best.pt"
LDM3D_CKPT = "runs/ldm3d/stage2_best.pt"
DATA_ROOT = "fmri"
DDIM_STEPS         = 50
TCDM_GUIDANCE_SCALE = 3.0
LDM3D_GUIDANCE_SCALE = 1.0

# ── helpers ───────────────────────────────────────────────────────────────────

def compute_metrics(pred: np.ndarray, target: np.ndarray, mask: np.ndarray):
    """All arrays: [D, H, W] float32, z-scored."""
    p, t = pred[mask], target[mask]
    mae  = float(np.abs(p - t).mean())
    mse  = float(((p - t) ** 2).mean())
    dr   = float(target.max() - target.min())
    ssim = structural_similarity(pred, target, data_range=dr)
    psnr = peak_signal_noise_ratio(target, pred, data_range=dr)
    return mae, mse, ssim, psnr


def accumulate(results, mae, mse, ssim, psnr):
    results["mae"].append(mae)
    results["mse"].append(mse)
    results["ssim"].append(ssim)
    results["psnr"].append(psnr)


def summarise(name, results):
    print(f"\n{name}")
    print(f"  MAE  : {np.mean(results['mae']):.5f} ± {np.std(results['mae']):.5f}")
    print(f"  MSE  : {np.mean(results['mse']):.5f} ± {np.std(results['mse']):.5f}")
    print(f"  SSIM : {np.mean(results['ssim']):.4f} ± {np.std(results['ssim']):.4f}")
    print(f"  PSNR : {np.mean(results['psnr']):.2f} ± {np.std(results['psnr']):.2f} dB")


def empty():
    return {"mae": [], "mse": [], "ssim": [], "psnr": []}

# ── main ─────────────────────────────────────────────────────────────────────

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp    = device.type == "cuda"

    # ── dataset ──────────────────────────────────────────────────────────────
    ds = PrePostFMRI(
        root_dir=DATA_ROOT,
        transform=ToChannelsFirstAndNormalize(nonzero_mask=True),
        strict=False,
        return_paths=True,
    )
    print(f"Dataset: {len(ds)} samples | device: {device}")

    r_tcdm, r_ldm3d, r_id = empty(), empty(), empty()

    # ── pass 1: 7TCDM-3D ─────────────────────────────────────────────────────
    print("\n--- 7TCDM-3D pass ---")
    from moyamoya.models.ldm_7tcdm3d import load_7tcdm3d_checkpoint
    tcdm, raw = load_7tcdm3d_checkpoint(TCDM_CKPT, device)   # resolves AE (embedded or referenced)
    for p in tcdm.parameters(): p.requires_grad_(False)

    for idx in range(len(ds)):
        x, y, meta = ds[idx]
        x_b = x.unsqueeze(0).to(device)
        y_b = y.unsqueeze(0).to(device)
        mask  = make_union_mask(x_b, y_b).squeeze().cpu().numpy().astype(bool)
        y_np  = y.squeeze().numpy()
        x_np  = x.squeeze().numpy()
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            pred = tcdm.generate(x_b, ddim_steps=DDIM_STEPS,
                                 guidance_scale=TCDM_GUIDANCE_SCALE)
        accumulate(r_tcdm, *compute_metrics(pred.squeeze().cpu().float().numpy(), y_np, mask))
        accumulate(r_id,   *compute_metrics(x_np, y_np, mask))
        print(f"[{idx+1:>3}/{len(ds)}] {meta['id']}  mae={r_tcdm['mae'][-1]:.4f}")

    del tcdm, raw
    torch.cuda.empty_cache()

    # ── pass 2: LDM-3D ───────────────────────────────────────────────────────
    print("\n--- LDM-3D pass ---")
    raw = torch.load(LDM3D_CKPT, map_location="cpu", weights_only=False)
    a   = raw["args"]
    ldm3d = build_paired_latent_diffusion(
        z_channels=a["z_channels"], embed_dim=a["embed_dim"],
        ae_ch=a["ae_ch"], ae_ch_mult=tuple(a.get("ae_ch_mult", [1,2,4])),
        ae_res_blocks=a["ae_res_blocks"], ae_resolution=a["ae_resolution"],
        kl_weight=a["kl_weight"],
        diff_base=a["diff_base"], t_dim=a["t_dim"], n_levels=a["n_levels"],
        T=a["T"],
    ).to(device)
    ldm3d.ae.load_state_dict(raw["ae"])
    ldm3d.denoiser.load_state_dict(raw["denoiser"])
    ldm3d.eval()
    for p in ldm3d.parameters(): p.requires_grad_(False)

    for idx in range(len(ds)):
        x, y, meta = ds[idx]
        x_b  = x.unsqueeze(0).to(device)
        y_b  = y.unsqueeze(0).to(device)
        mask = make_union_mask(x_b, y_b).squeeze().cpu().numpy().astype(bool)
        y_np = y.squeeze().numpy()
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            pred = ldm3d.generate(x_b, ddim_steps=DDIM_STEPS,
                                  guidance_scale=LDM3D_GUIDANCE_SCALE)
        accumulate(r_ldm3d, *compute_metrics(pred.squeeze().cpu().float().numpy(), y_np, mask))
        print(f"[{idx+1:>3}/{len(ds)}] {meta['id']}  mae={r_ldm3d['mae'][-1]:.4f}")

    del ldm3d, raw
    torch.cuda.empty_cache()

    # ── summary ──────────────────────────────────────────────────────────────
    print(f"\n{'='*55}")
    summarise("Identity baseline (pred = pre-surgery)", r_id)
    summarise("LDM-3D (stage-2 best)",                  r_ldm3d)
    summarise("7TCDM-3D (stage-2 best)",                r_tcdm)


if __name__ == "__main__":
    main()
