"""
Generate an ldm_7tcdm3d prediction for a single subject and export the full
predicted volume as a NIfTI (.nii.gz), reoriented back to the original geometry
(uses the source pre-surgery NIfTI's affine/header). Also writes a 3x3 PNG grid
(same renderer as eval_ldm_7tcdm3d) for visual verification.

The model builds entirely from the checkpoint's embedded `args`, so this works
for any variant (e.g. cfg3_rich_latent's z_channels=8/embed_dim=8).

NOTE on reproducibility: DDIM sampling starts from random latent noise. The
original eval (eval_ldm_7tcdm3d.py) does not seed, so its exact sample isn't
recoverable; this script seeds (--seed) so its own output is reproducible and
should match the eval result closely (same model + conditioning).

The exported volume is in the model's working space (per-volume z-score
normalised intensities), at the original voxel grid.

Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/predict_ldm_7tcdm3d_nii.py \
        --ckpt runs/cfg3_rich_latent/stage2_best.pt --subject 2024_040 \
        --out_dir outputs/predict/2024_040     # default: outputs/predict/<subject>
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import nibabel as nib

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from moyamoya.dataset import PrePostFMRI
from moyamoya.transform import ToChannelsFirstAndNormalize
from moyamoya.models.ldm_7tcdm3d import build_paired_latent_diffusion_7tcdm
from scripts.eval_ldm_7tcdm3d import _to_np, save_grid, compute_metrics


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt",           required=True)
    p.add_argument("--subject",        required=True, help="Subject id, e.g. 2024_040")
    p.add_argument("--data_root",      default="fmri")
    p.add_argument("--out_dir",        default=None)
    p.add_argument("--ddim_steps",     type=int,   default=None)
    p.add_argument("--guidance_scale", type=float, default=None)
    p.add_argument("--eta",            type=float, default=0.0)
    p.add_argument("--seed",           type=int,   default=42)
    p.add_argument("--n_samples",      type=int,   default=1,
                   help="Average this many independent DDIM draws (ensemble mean; >1 reduces sampling noise)")
    args = p.parse_args()

    out_dir = Path(args.out_dir or f"outputs/predict/{args.subject}")
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    raw = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    a = raw["args"]
    ddim_steps = args.ddim_steps     if args.ddim_steps     is not None else a.get("ddim_steps", 50)
    gs         = args.guidance_scale if args.guidance_scale is not None else a.get("guidance_scale", 3.0)

    model = build_paired_latent_diffusion_7tcdm(
        z_channels=a["z_channels"], embed_dim=a["embed_dim"], ae_ch=a["ae_ch"],
        ae_ch_mult=tuple(a["ae_ch_mult"]), ae_res_blocks=a["ae_res_blocks"],
        ae_resolution=a["ae_resolution"], kl_weight=a["kl_weight"],
        diff_dim=a["diff_dim"], diff_dim_mults=tuple(a["diff_dim_mults"]),
        init_kernel_size=a["init_kernel_size"], resnet_groups=a["resnet_groups"], T=a["T"],
    ).to(device)
    model.ae.load_state_dict(raw["ae"])
    model.denoiser.load_state_dict(raw["denoiser"])
    model.eval()
    print(f"Loaded {args.ckpt}  (ddim_steps={ddim_steps}, guidance_scale={gs}, "
          f"z_channels={a['z_channels']}, embed_dim={a['embed_dim']})")

    tfm = ToChannelsFirstAndNormalize(nonzero_mask=True)
    ds  = PrePostFMRI(args.data_root, transform=tfm, strict=False, return_paths=True)
    try:
        idx = next(i for i, (n, _, _) in enumerate(ds.pairs)
                   if n.replace(".nii.gz", "") == args.subject)
    except StopIteration:
        raise SystemExit(f"Subject {args.subject} not found in {args.data_root}")
    x, y, meta = ds[idx]
    x_b = x.unsqueeze(0).to(device)
    y_b = y.unsqueeze(0).to(device)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    with torch.no_grad():
        samples = [model.generate(x_b, ddim_steps=ddim_steps, eta=args.eta, guidance_scale=gs)
                   for _ in range(args.n_samples)]
    pred = torch.stack(samples, dim=0).mean(dim=0)   # averaging ensemble over independent DDIM draws

    mask = (x_b != 0) | (y_b != 0)
    m = compute_metrics(pred.float().cpu(), y_b, mask)
    print(f"{args.subject} (ensemble of {args.n_samples}): MAE={m['mae']:.4f}  "
          f"MSE={m['mse']:.4f}  PSNR={m['psnr']:.2f}  SSIM={m['ssim']:.4f}")

    # verification grid (same renderer / metrics as the eval output)
    png = out_dir / f"{args.subject}.png"
    save_grid(_to_np(x_b), _to_np(y_b), _to_np(pred), args.subject, m, png)

    # ── NIfTI export ──────────────────────────────────────────────────────────
    # transform did (X,Y,Z) -> permute(2,1,0) -> (Z,Y,X); invert it to (X,Y,Z).
    pred_zyx = pred.squeeze().cpu().float().numpy()              # (Z,Y,X)
    pred_xyz = np.ascontiguousarray(np.transpose(pred_zyx, (2, 1, 0))).astype(np.float32)
    pre_img  = nib.load(meta["pre_path"])
    if pred_xyz.shape != pre_img.shape:
        raise SystemExit(f"shape mismatch {pred_xyz.shape} vs source {pre_img.shape}")
    hdr = pre_img.header.copy()
    hdr.set_data_dtype(np.float32)
    out_nii = out_dir / f"{args.subject}_pred.nii.gz"
    nib.save(nib.Nifti1Image(pred_xyz, affine=pre_img.affine, header=hdr), out_nii)
    print(f"Saved volume : {out_nii}  shape={pred_xyz.shape}  (z-score normalised units)")
    print(f"Saved grid   : {png}")

    meta_path = out_dir / "metrics.json"
    with open(meta_path, "w") as f:
        json.dump({
            "model":          "ldm_7tcdm3d",
            "timestamp":      datetime.now().isoformat(),
            "subject":        args.subject,
            "ckpt":           args.ckpt,
            "ddim_steps":     ddim_steps,
            "guidance_scale": gs,
            "eta":            args.eta,
            "seed":           args.seed,
            "n_samples":      args.n_samples,
            "metrics":        {k: float(m[k]) for k in ("mae", "mse", "psnr", "ssim")},
        }, f, indent=2)
    print(f"Saved metrics: {meta_path}")


if __name__ == "__main__":
    main()
