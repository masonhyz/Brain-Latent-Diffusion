"""
TEMPORARY eval — metrics on a [0,1] scale for ldm_7tcdm3d.

For each validation subject:
  1. generate the prediction (same DDIM settings as the checkpoint),
  2. RESCALE each whole volume (prediction and target) from its z-score range to
     [0,1] via min-max:   v01 = (v - v.min()) / (v.max() - v.min())
     so the background (the per-volume minimum) -> 0 and the brightest voxel -> 1,
  3. (optional --zero_bg) force the background voxels of BOTH volumes to exactly 0,
  4. compute MAE / MSE / PSNR / SSIM with PSNR & SSIM using data_range = 1.0,
     over the brain mask (default) or the whole volume (--whole_volume).

Each volume is min-max rescaled by its OWN range (prediction and target separately).
Brain mask = union of non-background voxels (z-score background = per-volume min,
since raw CBF >= 0):   brain = (x != x.min()) | (y != y.min()).

Validation split is reconstructed exactly as in training (val_frac + seed from ckpt).

Usage:
  CUDA_VISIBLE_DEVICES=0 python scripts/eval_ldm_7tcdm3d_rescale01.py \
      --ckpt runs/cfg3_rich_latent/stage2_best.pt --whole_volume --zero_bg
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import random_split
from skimage.metrics import structural_similarity

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from moyamoya.dataset import PrePostFMRI
from moyamoya.transform import ToChannelsFirstAndNormalize
from moyamoya.models.ldm_7tcdm3d import build_paired_latent_diffusion_7tcdm


def _rescale01(v: np.ndarray) -> np.ndarray:
    """Min-max rescale a whole volume to [0,1]: min (background) -> 0, max -> 1."""
    lo, hi = float(v.min()), float(v.max())
    return (v - lo) / (hi - lo + 1e-8)


def metrics_rescale01(pred: torch.Tensor, target: torch.Tensor,
                      region: torch.Tensor, bg: torch.Tensor = None) -> dict:
    """
    MAE/MSE/PSNR/SSIM after min-max rescaling each volume to [0,1].
    region : boolean mask of voxels to score over (whole volume or brain).
    bg     : if given, background voxels forced to exactly 0 in BOTH volumes
             before scoring, so a matching background inflates the metrics.
    """
    p = _rescale01(pred.squeeze().cpu().float().numpy())
    t = _rescale01(target.squeeze().cpu().float().numpy())
    if bg is not None:
        b = bg.squeeze().cpu().bool().numpy()
        p[b] = 0.0
        t[b] = 0.0
    m = region.squeeze().cpu().bool().numpy()

    mp, mt = p[m], t[m]
    mae = float(np.abs(mp - mt).mean())
    mse = float(((mp - mt) ** 2).mean())
    psnr = float(10.0 * np.log10(1.0 / (mse + 1e-12)))          # data_range = 1
    _, smap = structural_similarity(t, p, data_range=1.0, win_size=7,
                                    channel_axis=None, full=True)
    ssim = float(smap[m].mean())
    return {"mae": mae, "mse": mse, "psnr": psnr, "ssim": ssim}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt",           default="runs/cfg3_rich_latent/stage2_best.pt")
    ap.add_argument("--data_root",      default="fmri")
    ap.add_argument("--ddim_steps",     type=int,   default=None)
    ap.add_argument("--guidance_scale", type=float, default=None)
    ap.add_argument("--eta",            type=float, default=0.0)
    ap.add_argument("--seed",           type=int,   default=None,
                    help="Seed for split + DDIM noise (default: ckpt's seed)")
    ap.add_argument("--out_csv",        default=None)
    ap.add_argument("--whole_volume",   action="store_true",
                    help="Score over the whole volume (no mask) instead of brain-masked")
    ap.add_argument("--zero_bg",        action="store_true",
                    help="Force background voxels (outside brain) to exactly 0 in both volumes")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raw = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    a = raw["args"]
    ddim_steps = args.ddim_steps     if args.ddim_steps     is not None else a.get("ddim_steps", 50)
    gs         = args.guidance_scale if args.guidance_scale is not None else a.get("guidance_scale", 3.0)
    seed       = args.seed           if args.seed           is not None else a.get("seed", 42)
    val_frac   = a.get("val_frac", 0.15)

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

    tfm = ToChannelsFirstAndNormalize(nonzero_mask=True)
    ds  = PrePostFMRI(args.data_root, transform=tfm, strict=False, return_paths=True)
    n_val   = max(1, int(len(ds) * val_frac))
    n_train = len(ds) - n_val
    g = torch.Generator().manual_seed(seed)
    _, val = random_split(ds, [n_train, n_val], generator=g)

    region_str = "WHOLE VOLUME (no mask)" if args.whole_volume else "brain-masked"
    if args.zero_bg:
        region_str += " + background forced to 0"
    print(f"[{args.ckpt}]  ddim_steps={ddim_steps} guidance_scale={gs} seed={seed}")
    print(f"Val subjects: {len(val)}  |  metric: per-volume MIN-MAX rescaled to [0,1], "
          f"{region_str}, data_range=1\n")

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    rows = []
    for j, idx in enumerate(val.indices):
        x, y, meta = ds[idx]
        x_b = x.unsqueeze(0).to(device)
        y_b = y.unsqueeze(0).to(device)
        with torch.no_grad():
            pred = model.generate(x_b, ddim_steps=ddim_steps, eta=args.eta, guidance_scale=gs)
        brain_anat = (x_b != x_b.min()) | (y_b != y_b.min())
        region = torch.ones_like(x_b, dtype=torch.bool) if args.whole_volume else brain_anat
        bg = (~brain_anat) if args.zero_bg else None
        m = metrics_rescale01(pred, y_b, region, bg=bg)
        rows.append({"id": meta["id"], **m})
        print(f"  [{j + 1:>2}/{len(val)}] {meta['id']}  "
              f"MAE={m['mae']:.4f}  MSE={m['mse']:.4f}  PSNR={m['psnr']:.2f}  SSIM={m['ssim']:.4f}")

    print("\n" + "=" * 60)
    print("[0,1] min-max rescaled, %s  (n=%d)" % (region_str, len(rows)))
    for k, lab in [("mae", "MAE"), ("mse", "MSE"), ("psnr", "PSNR(dB)"), ("ssim", "SSIM")]:
        v = [r[k] for r in rows]
        print(f"  {lab:9s} mean={np.mean(v):.5f}  std={np.std(v):.5f}  median={np.median(v):.5f}")
    tag = ("wholevol" if args.whole_volume else "val") + ("_zerobg" if args.zero_bg else "")
    out_csv = Path(args.out_csv or f"outputs/eval_cfg3_rescale01_{tag}/metrics.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", "mae", "mse", "psnr", "ssim"])
        w.writeheader(); w.writerows(rows)
    print(f"\nSaved: {out_csv}")


if __name__ == "__main__":
    main()
