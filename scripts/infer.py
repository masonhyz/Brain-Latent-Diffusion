"""
Paired diffusion inference: predict post-surgery CBF from pre-surgery CBF.

One entry point for all three paired models (replaces the former
infer_ldm.py / infer_ldm3d.py / infer_ldm_7tcdm3d.py):

    python scripts/infer.py --model ldm3d --ckpt runs/ldm3d/stage2_best.pt \\
        --data_root fmri --sample_idx 0 --ddim_steps 50 --out ldm3d_pred.png
"""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from moyamoya.dataset import PrePostFMRI
from moyamoya.transform import ToChannelsFirstAndNormalize
from moyamoya.utils import masked_l1, make_union_mask
from moyamoya.viz import save_ortho_grid


# ── per-model checkpoint loaders ──────────────────────────────────────────────

def _load_ldm(a, ckpt, device):
    from moyamoya.models.ldm import build_paired_diffusion
    model = build_paired_diffusion(
        base=a["diff_base"], t_dim=a["t_dim"], n_levels=a["n_levels"], T=a["T"],
    ).to(device)
    model.denoiser.load_state_dict(ckpt["denoiser"])
    return model


def _load_ldm3d(a, ckpt, device):
    from moyamoya.models.ldm3d import build_paired_latent_diffusion
    model = build_paired_latent_diffusion(
        z_channels=a.get("z_channels", 4),
        embed_dim=a.get("embed_dim", 4),
        ae_ch=a.get("ae_ch", 64),
        ae_res_blocks=a.get("ae_res_blocks", 2),
        ae_resolution=a.get("ae_resolution", 64),
        kl_weight=a.get("kl_weight", 1e-6),
        diff_base=a.get("diff_base", 64),
        t_dim=a.get("t_dim", 256),
        n_levels=a.get("n_levels", 3),
        T=a.get("T", 1000),
    ).to(device)
    model.ae.load_state_dict(ckpt["ae"])
    model.denoiser.load_state_dict(ckpt["denoiser"])
    return model


def _load_ldm_7tcdm3d(a, ckpt, device):
    from moyamoya.models.ldm_7tcdm3d import build_paired_latent_diffusion_7tcdm
    model = build_paired_latent_diffusion_7tcdm(
        z_channels=a.get("z_channels", 4),
        embed_dim=a.get("embed_dim", 4),
        ae_ch=a.get("ae_ch", 64),
        ae_res_blocks=a.get("ae_res_blocks", 2),
        ae_resolution=a.get("ae_resolution", 64),
        kl_weight=a.get("kl_weight", 1e-6),
        diff_dim=a.get("diff_dim", 32),
        diff_dim_mults=tuple(a.get("diff_dim_mults", [1, 2, 4, 8])),
        init_kernel_size=a.get("init_kernel_size", 3),
        resnet_groups=a.get("resnet_groups", 8),
        T=a.get("T", 1000),
    ).to(device)
    model.ae.load_state_dict(ckpt["ae"])
    model.denoiser.load_state_dict(ckpt["denoiser"])
    return model


# model name → (loader, figure title, default output filename)
MODELS = {
    "ldm":         (_load_ldm,         "PairedDiffusion", "ldm_prediction.png"),
    "ldm3d":       (_load_ldm3d,       "LDM3D",           "ldm3d_prediction.png"),
    "ldm_7tcdm3d": (_load_ldm_7tcdm3d, "LDM-7TCDM3D",     "ldm_7tcdm3d_prediction.png"),
}


def load_model(model_name: str, ckpt_path: str, device: torch.device):
    loader = MODELS[model_name][0]
    ckpt = torch.load(ckpt_path, map_location=device)
    model = loader(ckpt["args"], ckpt, device)
    model.eval()
    return model


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, choices=sorted(MODELS),
                   help="Which paired diffusion model the checkpoint belongs to")
    p.add_argument("--ckpt",           required=True, help="Path to the checkpoint")
    p.add_argument("--data_root",      default="fmri")
    p.add_argument("--sample_idx",     type=int,   default=0)
    p.add_argument("--ddim_steps",     type=int,   default=50)
    p.add_argument("--eta",            type=float, default=0.0,
                   help="DDIM stochasticity (0=deterministic, 1=DDPM)")
    p.add_argument("--guidance_scale", type=float, default=3.0,
                   help="CFG guidance scale (1.0 = no guidance; requires cfg_drop_prob > 0 at training)")
    p.add_argument("--out",            default=None)
    p.add_argument("--n_samples",      type=int,   default=1,
                   help="Number of samples to average (>1 explores stochasticity)")
    args = p.parse_args()

    title, default_out = MODELS[args.model][1], MODELS[args.model][2]
    out_path = args.out or default_out

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = load_model(args.model, args.ckpt, device)

    tfm = ToChannelsFirstAndNormalize(nonzero_mask=True)
    ds  = PrePostFMRI(root_dir=args.data_root, transform=tfm, strict=False)
    x, y = ds[args.sample_idx]
    x = x.unsqueeze(0).to(device)
    y = y.unsqueeze(0).to(device)
    print(f"Sample {args.sample_idx}: pre={tuple(x.shape)}  post={tuple(y.shape)}")

    with torch.no_grad():
        if args.n_samples == 1:
            y_pred = model.generate(x, ddim_steps=args.ddim_steps, eta=args.eta,
                                    guidance_scale=args.guidance_scale)
        else:
            preds = [model.generate(x, ddim_steps=args.ddim_steps, eta=args.eta,
                                    guidance_scale=args.guidance_scale)
                     for _ in range(args.n_samples)]
            y_pred = torch.stack(preds, dim=0).mean(dim=0)

    mask    = make_union_mask(x, y)
    loss    = masked_l1(y_pred, y, mask).item()
    id_loss = masked_l1(x, y, mask).item()
    print(f"Masked L1 (predicted vs real post): {loss:.4f}")
    print(f"Identity  (pre      vs real post):  {id_loss:.4f}")

    save_ortho_grid(x[0], y[0], y_pred[0], out_path,
                    title=f"{title} prediction  |  masked-L1 = {loss:.4f}")


if __name__ == "__main__":
    main()
