"""
Unified evaluation of all four models on the full paired dataset.

Models evaluated:
  1. 3D UNet          — runs/unet_prepost_bc128/best.pt
  2. 3D CycleGAN      — checkpoints/prepost_cyclegan3d/best_net_G_A.pth
  3. LDM-3D           — runs/ldm3d/stage2_best.pt
  4. 7TCDM-3D         — runs/ldm_7tcdm3d/stage2_best.pt

Metrics (computed in z-score normalized space, brain-masked):
  MAE, MSE, SSIM, PSNR

Usage (from project root):
    python scripts/eval_all_models.py
    python scripts/eval_all_models.py --ddim_steps 20  # faster LDM inference
"""

import sys
import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import nibabel as nib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from skimage.metrics import structural_similarity as ssim_fn
except ImportError:
    raise ImportError("pip install scikit-image")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
REPO_3D = ROOT / "third_party" / "3D-CycleGan-Pytorch-MedImaging"
sys.path.insert(0, str(REPO_3D))


# ── normalization (shared by UNet / LDM) ─────────────────────────────────────

def zscore_norm(vol: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Z-score over nonzero voxels, add channel dim → (1, D, H, W)."""
    mask = vol != 0
    if mask.any():
        vals = vol[mask]
        vol = (vol - vals.mean()) / vals.std(unbiased=False).clamp_min(eps)
    return vol.unsqueeze(0)          # (1, D, H, W)


def cyclegan_norm(vol: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Z-score → clamp(-3,3) → /3  →  (1,1,D,H,W)."""
    mask = vol > vol.max() * 0.01
    if mask.any():
        vals = vol[mask]
        vol = (vol - vals.mean()) / vals.std(unbiased=False).clamp_min(eps)
    vol = vol.clamp(-3.0, 3.0) / 3.0
    return vol.unsqueeze(0).unsqueeze(0)   # (1,1,D,H,W)


def load_raw(path: Path) -> torch.Tensor:
    return torch.from_numpy(nib.load(str(path)).get_fdata(dtype="float32"))


def pad8(vol: torch.Tensor) -> tuple[torch.Tensor, tuple]:
    """Pad to multiples of 8; return padded tensor + original shape."""
    orig = vol.shape[-3:]
    D, H, W = orig
    pD = (8 - D % 8) % 8
    pH = (8 - H % 8) % 8
    pW = (8 - W % 8) % 8
    if pD or pH or pW:
        vol = F.pad(vol, (0, pW, 0, pH, 0, pD), value=-1.0)
    return vol, orig


def unpad(vol: torch.Tensor, orig: tuple) -> torch.Tensor:
    D, H, W = orig
    return vol[..., :D, :H, :W]


# ── metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(pred: np.ndarray, target: np.ndarray,
                    mask: np.ndarray) -> dict:
    """
    pred, target, mask: 3D numpy arrays (D, H, W).
    Metrics computed over brain-masked voxels.
    SSIM/PSNR use the full volume with data_range = target range in mask.
    """
    p = pred[mask]
    t = target[mask]

    mae = float(np.abs(p - t).mean())
    mse = float(((p - t) ** 2).mean())

    data_range = float(target[mask].max() - target[mask].min())
    if data_range < 1e-8:
        data_range = 1.0

    psnr = float(10 * np.log10(data_range ** 2 / (mse + 1e-12)))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ssim_val = ssim_fn(
            target, pred,
            data_range=data_range,
            win_size=7,
            channel_axis=None,
        )

    return {"MAE": mae, "MSE": mse, "SSIM": ssim_val, "PSNR": psnr}


# ── model loaders ─────────────────────────────────────────────────────────────

def load_unet(ckpt_path: str, device: torch.device):
    from moyamoya.models.unet import UNet3D
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ck.get("model", ck)
    # infer base_channels from first conv
    first_w = next(v for k, v in state.items() if "weight" in k)
    base = first_w.shape[0]
    model = UNet3D(in_channels=1, out_channels=1, base=base).to(device)
    model.load_state_dict(state, strict=True)
    model.eval()
    print(f"  UNet loaded  (base={base})  from {ckpt_path}")
    return model


class _ResBlock3D(torch.nn.Module):
    """ResBlock matching the checkpoint's padding + Conv3d layout."""
    def __init__(self, dim: int):
        super().__init__()
        self.conv_block = torch.nn.Sequential(
            torch.nn.ReplicationPad3d(1),
            torch.nn.Conv3d(dim, dim, 3, bias=True),
            torch.nn.InstanceNorm3d(dim, affine=False, track_running_stats=False),
            torch.nn.ReLU(True),
            torch.nn.ReplicationPad3d(1),
            torch.nn.Conv3d(dim, dim, 3, bias=True),
            torch.nn.InstanceNorm3d(dim, affine=False, track_running_stats=False),
        )

    def forward(self, x):
        return x + self.conv_block(x)


def _build_cyclegan_generator(ngf: int, n_blocks: int) -> torch.nn.Module:
    """
    Reconstruct the exact ResNet generator architecture used at training time:
    ConvTranspose3d upsampling (not Upsample+Conv3d as in the current repo code).
    InstanceNorm affine=False, track_running_stats=False.
    """
    IN = lambda ch: torch.nn.InstanceNorm3d(ch, affine=False, track_running_stats=False)
    model = [
        torch.nn.ReplicationPad3d(3),
        torch.nn.Conv3d(1, ngf, 7, bias=True),
        IN(ngf), torch.nn.ReLU(True),
        torch.nn.Conv3d(ngf, ngf*2, 3, stride=2, padding=1, bias=True),
        IN(ngf*2), torch.nn.ReLU(True),
        torch.nn.Conv3d(ngf*2, ngf*4, 3, stride=2, padding=1, bias=True),
        IN(ngf*4), torch.nn.ReLU(True),
    ]
    for _ in range(n_blocks):
        model.append(_ResBlock3D(ngf*4))
    model += [
        torch.nn.ConvTranspose3d(ngf*4, ngf*2, 3, stride=2, padding=1,
                                  output_padding=1, bias=True),
        IN(ngf*2), torch.nn.ReLU(True),
        torch.nn.ConvTranspose3d(ngf*2, ngf, 3, stride=2, padding=1,
                                  output_padding=1, bias=True),
        IN(ngf), torch.nn.ReLU(True),
        torch.nn.ReplicationPad3d(3),
        torch.nn.Conv3d(ngf, 1, 7, bias=True),
        torch.nn.Tanh(),
    ]
    return torch.nn.Sequential(*model)


def load_cyclegan(ckpt_dir: str, device: torch.device):
    ckpt_dir = Path(ckpt_dir)
    G_path = ckpt_dir / "best_net_G_A.pth"
    sd = torch.load(str(G_path), map_location="cpu", weights_only=False)
    ngf = sd["model.1.weight"].shape[0]
    n_blocks = len({k.split(".")[1] for k in sd if "conv_block" in k})

    gen = _build_cyclegan_generator(ngf, n_blocks)
    # wrap in nn.Module so state_dict has "model." prefix
    class _G(torch.nn.Module):
        def __init__(self, seq): super().__init__(); self.model = seq
        def forward(self, x): return self.model(x)

    model = _G(gen).to(device)
    model.load_state_dict(sd, strict=True)
    model.eval()
    print(f"  CycleGAN loaded  (ngf={ngf}, {n_blocks} resblocks)  from {G_path}")
    return model


def load_ldm3d(ckpt_path: str, device: torch.device):
    from moyamoya.models.ldm3d import build_paired_latent_diffusion
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    a = ck["args"]
    model = build_paired_latent_diffusion(
        z_channels=a["z_channels"], embed_dim=a["embed_dim"],
        ae_ch=a["ae_ch"], ae_ch_mult=tuple(a.get("ae_ch_mult", [1, 2, 4])),
        ae_res_blocks=a["ae_res_blocks"], ae_resolution=a["ae_resolution"],
        kl_weight=a["kl_weight"],
        diff_base=a["diff_base"], t_dim=a["t_dim"], n_levels=a["n_levels"],
        T=a["T"],
    ).to(device)
    model.ae.load_state_dict(ck["ae"])
    model.denoiser.load_state_dict(ck["denoiser"])
    model.eval()
    print(f"  LDM-3D loaded  from {ckpt_path}")
    return model, a.get("ddim_steps", 50), a.get("guidance_scale", 1.0)


def load_7tcdm(ckpt_path: str, device: torch.device):
    from moyamoya.models.ldm_7tcdm3d import build_paired_latent_diffusion_7tcdm
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    a = ck["args"]
    model = build_paired_latent_diffusion_7tcdm(
        z_channels=a["z_channels"], embed_dim=a["embed_dim"],
        ae_ch=a["ae_ch"], ae_ch_mult=tuple(a.get("ae_ch_mult", [1, 2, 4])),
        ae_res_blocks=a["ae_res_blocks"], ae_resolution=a["ae_resolution"],
        kl_weight=a["kl_weight"],
        diff_dim=a["diff_dim"], diff_dim_mults=tuple(a["diff_dim_mults"]),
        init_kernel_size=a.get("init_kernel_size", 3),
        resnet_groups=a.get("resnet_groups", 8),
        T=a["T"],
    ).to(device)
    model.ae.load_state_dict(ck["ae"])
    model.denoiser.load_state_dict(ck["denoiser"])
    model.eval()
    print(f"  7TCDM-3D loaded  from {ckpt_path}")
    return model, a.get("ddim_steps", 50), a.get("guidance_scale", 3.0)


# ── per-model inference ───────────────────────────────────────────────────────

@torch.no_grad()
def infer_unet(model, x_raw: torch.Tensor, device: torch.device) -> np.ndarray:
    x = zscore_norm(x_raw).unsqueeze(0).to(device)          # (1,1,D,H,W)
    x, orig = pad8(x)
    pred = model(x)
    pred = unpad(pred, orig).squeeze().cpu().numpy()
    return pred


@torch.no_grad()
def infer_cyclegan(model, x_raw: torch.Tensor, device: torch.device) -> np.ndarray:
    x = cyclegan_norm(x_raw).to(device)                      # (1,1,D,H,W)
    x, orig = pad8(x)
    pred = model(x)
    pred = unpad(pred, orig)
    pred = pred.squeeze().cpu().numpy() * 3.0                # tanh→zscore
    return pred


@torch.no_grad()
def infer_ldm(model, x_raw: torch.Tensor, device: torch.device,
              ddim_steps: int, guidance_scale: float) -> np.ndarray:
    x = zscore_norm(x_raw).unsqueeze(0).to(device)          # (1,1,D,H,W)
    pred = model.generate(x, ddim_steps=ddim_steps,
                          guidance_scale=guidance_scale)
    pred = pred.squeeze().cpu().numpy()
    return pred


# ── main evaluation loop ──────────────────────────────────────────────────────

def evaluate_model(name: str, infer_fn, pre_files, post_map, device) -> list[dict]:
    results = []
    n = len(pre_files)
    for i, pre_path in enumerate(pre_files):
        subj = pre_path.name
        post_path = post_map.get(subj)
        if post_path is None:
            continue

        x_raw = load_raw(pre_path)
        y_raw = load_raw(post_path)

        try:
            pred = infer_fn(x_raw)
        except Exception as e:
            print(f"  [{name}] SKIP {subj}: {e}")
            continue

        # ground-truth in z-score space (same as UNet/LDM output space)
        target = zscore_norm(y_raw).squeeze().numpy()   # (D,H,W)
        brain_mask = (y_raw.numpy() > 0)

        m = compute_metrics(pred, target, brain_mask)
        m["subject"] = subj
        results.append(m)

        if (i + 1) % 50 == 0 or (i + 1) == n:
            running_mae = np.mean([r["MAE"] for r in results])
            print(f"  [{name}] {i+1}/{n}  running MAE={running_mae:.4f}")

    return results


# ── plotting ──────────────────────────────────────────────────────────────────

def plot_comparison(summary: dict, out_path: Path):
    models  = list(summary.keys())
    metrics = ["MAE", "MSE", "SSIM", "PSNR"]
    n_models  = len(models)
    n_metrics = len(metrics)

    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

    fig, axes = plt.subplots(1, n_metrics, figsize=(4.5 * n_metrics, 5))

    for ax, metric in zip(axes, metrics):
        vals  = [summary[m][metric]["mean"] for m in models]
        stds  = [summary[m][metric]["std"]  for m in models]

        x = np.arange(n_models)
        bars = ax.bar(x, vals, yerr=stds, capsize=4,
                      color=colors[:n_models], width=0.55,
                      error_kw={"elinewidth": 1.2, "alpha": 0.8})

        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=20, ha="right", fontsize=10)
        ax.set_title(metric, fontsize=13, fontweight="bold")
        ax.set_ylabel(metric, fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)

        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(stds) * 0.05,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=8.5)

    fig.suptitle("Model Comparison — Full Dataset (brain-masked, z-score space)",
                 fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"\nChart saved: {out_path}")


def print_table(summary: dict):
    metrics = ["MAE", "MSE", "SSIM", "PSNR"]
    col_w = 18
    print("\n" + "=" * (12 + col_w * len(metrics)))
    print(f"{'Model':<12}" + "".join(f"{m:>{col_w}}" for m in metrics))
    print("-" * (12 + col_w * len(metrics)))
    for model, stats in summary.items():
        row = f"{model:<12}"
        for m in metrics:
            mean = stats[m]["mean"]
            std  = stats[m]["std"]
            row += f"{mean:>10.4f}±{std:<6.4f}"
        print(row)
    print("=" * (12 + col_w * len(metrics)))


# ── entry point ───────────────────────────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root",       default="fmri")
    p.add_argument("--unet_ckpt",       default="runs/unet_prepost_bc128/best.pt")
    p.add_argument("--cyclegan_dir",    default="checkpoints/prepost_cyclegan3d")
    p.add_argument("--ldm3d_ckpt",      default="runs/ldm3d/stage2_best.pt")
    p.add_argument("--tcdm_ckpt",       default="runs/ldm_7tcdm3d/stage2_best.pt")
    p.add_argument("--out_dir",         default="outputs/eval_comparison")
    p.add_argument("--ddim_steps",      type=int, default=50)
    return p.parse_args()


def main():
    args   = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pre_dir  = Path(args.data_root) / "pre_surgery"
    post_dir = Path(args.data_root) / "6_months_post_surgery"
    pre_files = sorted(pre_dir.glob("*.nii.gz"))
    post_map  = {p.name: p for p in post_dir.glob("*.nii.gz")}
    print(f"Dataset: {len(pre_files)} subjects\n")

    all_results = {}

    # ── UNet ─────────────────────────────────────────────────────────────────
    print("Loading UNet…")
    unet = load_unet(args.unet_ckpt, device)
    all_results["UNet-3D"] = evaluate_model(
        "UNet-3D",
        lambda x: infer_unet(unet, x, device),
        pre_files, post_map, device,
    )
    del unet
    torch.cuda.empty_cache()

    # ── CycleGAN ─────────────────────────────────────────────────────────────
    print("\nLoading CycleGAN…")
    cgan = load_cyclegan(args.cyclegan_dir, device)
    all_results["CycleGAN-3D"] = evaluate_model(
        "CycleGAN-3D",
        lambda x: infer_cyclegan(cgan, x, device),
        pre_files, post_map, device,
    )
    del cgan
    torch.cuda.empty_cache()

    # ── LDM-3D ───────────────────────────────────────────────────────────────
    print("\nLoading LDM-3D…")
    ldm, ldm_steps, ldm_gs = load_ldm3d(args.ldm3d_ckpt, device)
    ddim_steps = args.ddim_steps or ldm_steps
    all_results["LDM-3D"] = evaluate_model(
        "LDM-3D",
        lambda x: infer_ldm(ldm, x, device, ddim_steps, ldm_gs),
        pre_files, post_map, device,
    )
    del ldm
    torch.cuda.empty_cache()

    # ── 7TCDM-3D ─────────────────────────────────────────────────────────────
    print("\nLoading 7TCDM-3D…")
    tcdm, tcdm_steps, tcdm_gs = load_7tcdm(args.tcdm_ckpt, device)
    all_results["7TCDM-3D"] = evaluate_model(
        "7TCDM-3D",
        lambda x: infer_ldm(tcdm, x, device, args.ddim_steps or tcdm_steps, tcdm_gs),
        pre_files, post_map, device,
    )
    del tcdm
    torch.cuda.empty_cache()

    # ── aggregate ─────────────────────────────────────────────────────────────
    metrics  = ["MAE", "MSE", "SSIM", "PSNR"]
    summary  = {}
    for model_name, records in all_results.items():
        summary[model_name] = {}
        for m in metrics:
            vals = np.array([r[m] for r in records])
            summary[model_name][m] = {"mean": float(vals.mean()),
                                      "std":  float(vals.std())}

    print_table(summary)
    plot_comparison(summary, out_dir / "model_comparison.png")

    # Save raw per-subject results
    with open(out_dir / "per_subject_results.json", "w") as f:
        json.dump({k: v for k, v in all_results.items()}, f, indent=2)
    print(f"Per-subject results: {out_dir / 'per_subject_results.json'}")


if __name__ == "__main__":
    main()
