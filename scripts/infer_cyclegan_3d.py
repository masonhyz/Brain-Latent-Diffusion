"""
infer_fmri_3d_cyclegan.py
─────────────────────────
Post-hoc visualization of 3D CycleGAN results.

Loads the latest (or a specified epoch) generator from a checkpoint directory,
runs inference on the full dataset, and saves a 3×3-grid HTML exactly like the
one produced during training:

    cols: real_A (pre)  |  fake_B (generated post)  |  real_B (real post)
    rows: sagittal / axial / coronal

Architecture config (ngf, netG) is auto-detected from the saved weights so the
model is always reconstructed correctly regardless of what flags were used at
training time.

Usage:
    # All subjects, latest checkpoint:
    python infer_fmri_3d_cyclegan.py

    # Specific epoch:
    python infer_fmri_3d_cyclegan.py --which_epoch 20

    # Every 10th subject only:
    python infer_fmri_3d_cyclegan.py --viz_every 10
"""

from __future__ import annotations

import sys
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
import nibabel as nib

# ── 3D CycleGAN repo ─────────────────────────────────────────────────────────
HERE    = Path(__file__).resolve().parent
REPO_3D = HERE.parent / "third_party" / "3D-CycleGan-Pytorch-MedImaging"
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(REPO_3D))

from models.cycle_gan_model import CycleGANModel  # noqa: E402
from moyamoya.viz.html_viz import VolumeVisualizer  # noqa: E402


# ─────────────────────────────────────────────────────────────────────────────
# Auto-detect architecture from saved weights
# ─────────────────────────────────────────────────────────────────────────────

def detect_arch(ckpt_path: Path) -> tuple[int, str]:
    """
    Infer ngf and netG from the generator checkpoint weights.

    The 3D ResNet generator layout is:
        model.1.weight  shape (ngf, in_nc, 7,7,7)   ← first conv
        model.10 … model.10+(n_blocks-1)             ← resnet blocks
    Returns (ngf, netG_string).
    """
    sd = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)

    # ngf = output channels of the first conv
    ngf = sd["model.1.weight"].shape[0]

    # number of resnet blocks = number of distinct indices that have 'conv_block'
    res_indices = {k.split(".")[1] for k in sd if "conv_block" in k}
    n_blocks = len(res_indices)
    if n_blocks == 9:
        netG = "resnet_9blocks"
    elif n_blocks == 6:
        netG = "resnet_6blocks"
    else:
        raise ValueError(
            f"Unexpected number of resnet blocks ({n_blocks}) in {ckpt_path.name}; "
            "expected 6 or 9."
        )

    return ngf, netG


# ─────────────────────────────────────────────────────────────────────────────
# Volume helpers  (mirrors FMRIDataset3D exactly)
# ─────────────────────────────────────────────────────────────────────────────

def load_raw(path: Path) -> torch.Tensor:
    img = nib.load(str(path))
    return torch.from_numpy(img.get_fdata(dtype="float32"))


def normalize(vol: torch.Tensor, mode: str = "zscore") -> torch.Tensor:
    if mode == "zscore":
        mask = vol > vol.max() * 0.01
        if mask.any():
            vals = vol[mask]
            vol  = (vol - vals.mean()) / vals.std(unbiased=False).clamp_min(1e-6)
        return vol.clamp(-3.0, 3.0) / 3.0
    elif mode == "minmax":
        lo, hi = vol.min(), vol.max()
        if hi > lo:
            return (vol - lo) / (hi - lo) * 2.0 - 1.0
        return torch.zeros_like(vol)
    else:
        raise ValueError(f"Unknown norm_mode: {mode}")


def pad_to_divisor(vol: torch.Tensor, d: int = 8) -> torch.Tensor:
    D, H, W = vol.shape
    pD = (d - D % d) % d
    pH = (d - H % d) % d
    pW = (d - W % d) % d
    if pD or pH or pW:
        vol = F.pad(vol, (0, pW, 0, pH, 0, pD), mode="constant", value=-1.0)
    return vol


def load_volume(path: Path, norm_mode: str = "zscore") -> torch.Tensor:
    """Load → normalize → pad → (1,1,D,H,W) tensor."""
    vol = load_raw(path)
    vol = normalize(vol, norm_mode)
    vol = pad_to_divisor(vol)
    return vol.unsqueeze(0).unsqueeze(0)  # (1,1,D,H,W)


# ─────────────────────────────────────────────────────────────────────────────
# Model setup
# ─────────────────────────────────────────────────────────────────────────────

def build_opt(args: argparse.Namespace, ngf: int, netG: str) -> argparse.Namespace:
    import argparse as _ap
    opt = _ap.Namespace()

    opt.checkpoints_dir = args.checkpoints_dir
    opt.name            = args.name

    if torch.cuda.is_available() and args.gpu_ids:
        opt.gpu_ids = [int(x) for x in args.gpu_ids.split(",")]
    else:
        opt.gpu_ids = []

    # Architecture — sourced from checkpoint, not CLI defaults
    opt.input_nc   = 1
    opt.output_nc  = 1
    opt.ngf        = ngf
    opt.ndf        = 64       # discriminator not loaded in inference; value irrelevant
    opt.netG       = netG
    opt.netD       = "n_layers"
    opt.n_layers_D = 3
    opt.norm       = "instance"
    opt.no_dropout = True
    opt.init_type  = "normal"
    opt.init_gain  = 0.02

    # Loss weights — only needed to satisfy model.__init__; not used during inference
    opt.lambda_A        = 10.0
    opt.lambda_B        = 10.0
    opt.lambda_identity = 0.5
    opt.lambda_co_A     = 0.0
    opt.lambda_co_B     = 0.0
    opt.no_lsgan        = False
    opt.pool_size       = 50

    # Scheduler params — not used during inference
    opt.beta1          = 0.5
    opt.lr             = 2e-4
    opt.lr_policy      = "lambda"
    opt.epoch_count    = 1
    opt.niter          = 100
    opt.niter_decay    = 100
    opt.lr_decay_iters = 50

    opt.isTrain         = False
    opt.continue_train  = False
    opt.which_epoch     = args.which_epoch
    opt.which_direction = "AtoB"
    opt.verbose         = False
    opt.model           = "cycle_gan"

    return opt


# ─────────────────────────────────────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def infer_pair(model: CycleGANModel, A: torch.Tensor, B: torch.Tensor) -> dict:
    device = model.device
    model.set_input((A.to(device), B.to(device)))
    model.forward()
    return model.get_current_visuals()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Post-hoc 3D CycleGAN inference + HTML visualization",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--dataroot",        default="fmri")
    p.add_argument("--pre_dirname",     default="pre_surgery")
    p.add_argument("--post_dirname",    default="6_months_post_surgery")
    p.add_argument("--norm_mode",       default="zscore", choices=["zscore", "minmax"])
    p.add_argument("--name",            default="fmri_3d_cyclegan3")
    p.add_argument("--checkpoints_dir", default="checkpoints")
    p.add_argument("--which_epoch",     default="latest",
                   help="'latest' or an epoch number (e.g. 20)")
    p.add_argument("--gpu_ids",         default="0")
    p.add_argument("--viz_every",       type=int, default=1,
                   help="Save a snapshot every N volumes (1 = all)")
    p.add_argument("--out_web_dir",     default=None,
                   help="Override output dir (default: <ckpt>/web_infer)")
    return p


def main():
    parser = build_parser()
    args   = parser.parse_args()

    ckpt_dir = Path(args.checkpoints_dir) / args.name
    if not ckpt_dir.is_dir():
        raise FileNotFoundError(f"Checkpoint directory not found: {ckpt_dir}")

    # ── detect architecture from saved weights ────────────────────────────
    epoch_tag = args.which_epoch
    G_A_path  = ckpt_dir / f"{epoch_tag}_net_G_A.pth"
    if not G_A_path.exists():
        raise FileNotFoundError(f"Generator checkpoint not found: {G_A_path}")

    ngf, netG = detect_arch(G_A_path)
    print(f"[INFO] Detected from checkpoint: ngf={ngf}, netG={netG}")

    # ── output directory ──────────────────────────────────────────────────
    web_dir = Path(args.out_web_dir) if args.out_web_dir else ckpt_dir / "web_infer"
    viz = VolumeVisualizer(
        web_dir      = web_dir,
        title        = f"3D CycleGAN Inference — {args.name} ({epoch_tag})",
        refresh_secs = 0,
    )
    print(f"[INFO] HTML output: {web_dir / 'index.html'}")

    # ── build model ───────────────────────────────────────────────────────
    opt   = build_opt(args, ngf=ngf, netG=netG)
    model = CycleGANModel()
    model.initialize(opt)
    model.setup(opt)
    model.eval()
    print(f"[INFO] Loaded {epoch_tag} checkpoint  |  device: {model.device}")

    # ── collect volumes ───────────────────────────────────────────────────
    pre_dir   = Path(args.dataroot) / args.pre_dirname
    post_dir  = Path(args.dataroot) / args.post_dirname
    pre_files = sorted(pre_dir.glob("*.nii.gz"))
    post_map  = {p.name: p for p in sorted(post_dir.glob("*.nii.gz"))}

    if not pre_files:
        raise FileNotFoundError(f"No .nii.gz files found in {pre_dir}")

    viz_every = max(1, args.viz_every)
    n_snaps   = sum(1 for i in range(len(pre_files)) if i % viz_every == 0)
    print(f"[INFO] {len(pre_files)} pre-surgery volumes  |  "
          f"viz every {viz_every} → {n_snaps} snapshots")

    post_list = sorted(post_dir.glob("*.nii.gz"))   # positional fallback

    for idx, pre_path in enumerate(pre_files):
        if idx % viz_every != 0:
            continue

        subj = pre_path.name
        if subj in post_map:
            post_path = post_map[subj]
        else:
            post_path = post_list[idx % len(post_list)]
            print(f"  [WARN] No filename match for {subj}; using {post_path.name}")

        print(f"[{idx+1:3d}/{len(pre_files)}] {subj} ↔ {post_path.name}", end=" … ", flush=True)

        A = load_volume(pre_path,  args.norm_mode)
        B = load_volume(post_path, args.norm_mode)

        visuals = infer_pair(model, A, B)
        viz.save_volumes(epoch=0, iters=idx + 1, volumes=visuals)
        print("saved")

    print(f"\n[DONE] HTML visualization at: {web_dir / 'index.html'}")


if __name__ == "__main__":
    main()
