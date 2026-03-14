#!/usr/bin/env python3
"""
train_fmri_cyclegan_2d.py
─────────────────────────
Convenience launcher for training a 2D image-to-image model on fMRI slices.

Supports two model modes:

  pix2pix  (--model pix2pix --paired, RECOMMENDED for pre/post fMRI)
      Supervised paired training.  Each pre-surgery slice is matched to the
      post-surgery slice of the *same subject and same z-index*.  The L1 loss
      directly penalises pixel-level deviation from the real post-surgery scan,
      giving unambiguous gradient signal about what changed per subject.
      The conditional discriminator asks "is this a plausible change from THIS
      input", not just "does this look like a post scan in general".

  cycle_gan  (--model cycle_gan, unpaired)
      Unsupervised domain translation.  Designed for domains with a large,
      domain-level gap (e.g. horse ↔ zebra).  For pre/post fMRI the gap is
      too small — the discriminator cannot reliably distinguish the domains,
      so it provides near-zero gradient and the generator defaults to identity.
      Only useful if you truly have no subject-level pairing.

Delegates entirely to pytorch-CycleGAN-and-pix2pix/train.py – no model
code is duplicated here.  The repo's built-in Visualizer writes an
auto-refreshing HTML page showing generated slices during training:

    <checkpoints_dir>/<name>/web/index.html

Usage
─────
    # Recommended: paired pix2pix
    python train_fmri_cyclegan_2d.py --model pix2pix --name fmri_pix2pix

    # Unpaired CycleGAN (use only if data is truly unpaired)
    python train_fmri_cyclegan_2d.py --model cycle_gan --name fmri_cyclegan

    # Override any default, or pass extra train.py flags after --:
    python train_fmri_cyclegan_2d.py --model pix2pix --name my_run \\
        -- --lambda_L1 200

Key flags
─────────
  --model           pix2pix (default, recommended) | cycle_gan
  --name            Experiment name; used for checkpoint & HTML directories.
  --dataroot        fMRI root folder (must contain pre_surgery/ and
                    6_months_post_surgery/ subdirectories).
  --slice_size      Spatial resolution of 2D slices fed to the network.
  --netG            Generator: resnet_9blocks (default) | unet_256 | unet_128
  --lambda_L1       pix2pix L1 weight (default 100; increase for sharper output)
  --n_epochs        Epochs at the initial learning rate.
  --n_epochs_decay  Epochs with linear LR decay.
"""

import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent / "third_party" / "pytorch-CycleGAN-and-pix2pix"
FMRI_ROOT = HERE.parent / "fmri"
CHECKPOINTS = HERE.parent / "checkpoints"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Launch 2D fMRI image-to-image training via pytorch-CycleGAN-and-pix2pix",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # ── model mode ──────────────────────────────────────────────────────
    p.add_argument("--model",           default="pix2pix",
                   choices=["pix2pix", "cycle_gan"],
                   help="pix2pix = supervised paired training (recommended for pre/post fMRI); "
                        "cycle_gan = unsupervised unpaired training")

    # ── experiment ──────────────────────────────────────────────────────
    p.add_argument("--name",            default="fmri_pix2pix",
                   help="Experiment name; HTML log at <checkpoints_dir>/<name>/web/index.html")
    p.add_argument("--dataroot",        default=str(FMRI_ROOT),
                   help="Root fMRI directory (pre_surgery/ and 6_months_post_surgery/ inside)")
    p.add_argument("--checkpoints_dir", default=str(CHECKPOINTS))

    # ── hardware ────────────────────────────────────────────────────────
    p.add_argument("--batch_size",      type=int, default=4)
    p.add_argument("--num_threads",     type=int, default=4,
                   help="DataLoader worker threads")

    # ── training schedule ───────────────────────────────────────────────
    p.add_argument("--n_epochs",        type=int, default=100,
                   help="Epochs at the initial learning rate")
    p.add_argument("--n_epochs_decay",  type=int, default=100,
                   help="Epochs with linear LR decay to zero")
    p.add_argument("--lr",              type=float, default=2e-4,
                   help="Initial Adam learning rate")

    # ── network ─────────────────────────────────────────────────────────
    p.add_argument("--netG",            default="resnet_9blocks",
                   choices=["resnet_9blocks", "resnet_6blocks", "unet_256", "unet_128"],
                   help="Generator architecture")
    p.add_argument("--ndf",             type=int, default=64,
                   help="Discriminator base filter count")

    # ── pix2pix loss weights ─────────────────────────────────────────────
    # lambda_L1: the reconstruction loss ||fake_B - real_B||_1.
    # This is the signal that actually teaches the generator what changed
    # per subject.  Higher = sharper / more faithful to real_B.
    # 100 is the pix2pix paper default; 200 can help for subtle changes.
    p.add_argument("--lambda_L1",       type=float, default=100.0,
                   help="pix2pix L1 reconstruction loss weight. "
                        "This directly penalises per-pixel deviation from real_B. "
                        "Increase to 200 if outputs are too blurry.")

    # ── cycle_gan loss weights (only used when --model cycle_gan) ────────
    # lambda_identity=0.1: small regulariser that prevents the negation trick
    #   (G_A(x)=-x is cycle-consistent but identity loss penalises it).
    # lambda_A/B=1: very low cycle weight gives the adversarial signal more
    #   leverage; high lambda lets identity trivially win for similar domains.
    p.add_argument("--lambda_identity", type=float, default=0.1,
                   help="[cycle_gan only] Identity loss weight. "
                        "0.1 prevents negation collapse without promoting identity.")
    p.add_argument("--lambda_A",        type=float, default=1.0,
                   help="[cycle_gan only] Cycle-consistency weight for A→B→A")
    p.add_argument("--lambda_B",        type=float, default=1.0,
                   help="[cycle_gan only] Cycle-consistency weight for B→A→B")

    # ── fMRI-slice dataset ───────────────────────────────────────────────
    p.add_argument("--slice_size",      type=int, default=256,
                   help="Spatial size (square) slices are resized to before the network")
    p.add_argument("--pre_dirname",     default="pre_surgery",
                   help="Subdirectory name for pre-surgery volumes under dataroot")
    p.add_argument("--post_dirname",    default="6_months_post_surgery",
                   help="Subdirectory name for post-surgery volumes under dataroot")
    p.add_argument("--norm_mode",      default="zscore",
                   choices=["zscore", "minmax", "none"],
                   help="'zscore' (default): brain-masked z-score + clip ±3σ + [-1,1]; "
                        "'minmax': min-max to [-1,1], no z-scoring; "
                        "'none': raw values (WARNING: produces garbage with tanh output).")
    p.add_argument("--norm_scope",     default="volume",
                   choices=["volume", "dataset"],
                   help="'volume' (default): normalize each scan independently (erases brightness differences); "
                        "'dataset': joint stats across all pre+post volumes — preserves systematic "
                        "brightness differences between pre and post.")
    p.add_argument("--no_normalize",   action="store_true", default=False,
                   help="Deprecated alias for --norm_mode none.")

    # ── replay buffer ────────────────────────────────────────────────────
    p.add_argument("--pool_size",       type=int, default=0,
                   help="GAN image replay buffer size (0 = disabled; recommended for similar domains)")

    # ── visualisation ───────────────────────────────────────────────────
    p.add_argument("--display_freq",    type=int, default=400,
                   help="Save generated slice images to HTML every N iterations")
    p.add_argument("--update_html_freq", type=int, default=1000,
                   help="Rebuild HTML page every N iterations")
    p.add_argument("--print_freq",      type=int, default=100,
                   help="Print losses to console every N iterations")

    return p


def main():
    parser = build_parser()
    known, extra = parser.parse_known_args()

    if known.no_normalize:
        print("[WARN] --no_normalize is deprecated; use --norm_mode none instead. Mapping automatically.")
        known.norm_mode = "none"

    is_pix2pix = (known.model == "pix2pix")
    web_index  = Path(known.checkpoints_dir) / known.name / "web" / "index.html"

    cmd = [
        sys.executable, str(REPO / "train.py"),
        # ── data ──────────────────────────────────────────────────────────
        "--dataroot",           known.dataroot,
        "--dataset_mode",       "fmri_slice",
        "--pre_dirname",        known.pre_dirname,
        "--post_dirname",       known.post_dirname,
        "--slice_size",         str(known.slice_size),
        # ── model ─────────────────────────────────────────────────────────
        "--model",              known.model,
        "--input_nc",           "1",
        "--output_nc",          "1",
        "--netG",               known.netG,
        "--ndf",                str(known.ndf),
        "--netD",               "basic",
        "--norm",               "instance",
        # ── training ──────────────────────────────────────────────────────
        "--n_epochs",           str(known.n_epochs),
        "--n_epochs_decay",     str(known.n_epochs_decay),
        "--lr",                 str(known.lr),
        "--batch_size",         str(known.batch_size),
        "--num_threads",        str(known.num_threads),
        "--pool_size",          str(known.pool_size),
        # ── experiment & logging ──────────────────────────────────────────
        "--name",               known.name,
        "--checkpoints_dir",    known.checkpoints_dir,
        "--display_freq",       str(known.display_freq),
        "--update_html_freq",   str(known.update_html_freq),
        "--print_freq",         str(known.print_freq),
    ]

    cmd += ["--norm_mode", known.norm_mode, "--norm_scope", known.norm_scope]

    if is_pix2pix:
        # pix2pix: use paired sampling and L1 reconstruction loss.
        # The --paired flag is passed to fmri_slice_dataset.
        cmd += [
            "--paired",
            "--lambda_L1",      str(known.lambda_L1),
            "--direction",      "AtoB",
        ]
    else:
        # cycle_gan: unpaired adversarial training with cycle consistency.
        cmd += [
            "--lambda_A",       str(known.lambda_A),
            "--lambda_B",       str(known.lambda_B),
            "--lambda_identity", str(known.lambda_identity),
        ]

    cmd += [a for a in extra if a.strip()]   # any extra train.py flags passed after the above

    print("=" * 62)
    print(f"  Model         : {known.model}")
    print(f"  Experiment    : {known.name}")
    print(f"  Data root     : {known.dataroot}")
    print(f"  Slices        : axial only, middle 60%% of Z, size={known.slice_size}px")
    if is_pix2pix:
        print(f"  Paired        : True (same subject + z per sample)")
        print(f"  λ_L1          : {known.lambda_L1}")
    else:
        print(f"  Paired        : False (random B per sample)")
        print(f"  λ_cycle (A/B) : {known.lambda_A} / {known.lambda_B}")
        print(f"  λ_identity    : {known.lambda_identity}")
    print(f"  ndf           : {known.ndf}  pool_size={known.pool_size}")
    print(f"  HTML log      : {web_index}")
    print("=" * 62)
    print("Command:\n  " + " \\\n    ".join(cmd) + "\n")

    subprocess.run(cmd, cwd=str(REPO), check=True)


if __name__ == "__main__":
    main()
