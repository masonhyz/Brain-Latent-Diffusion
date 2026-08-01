"""
DEPRECATED — merged into eval_ldm_7tcdm3d.py.

Single-subject prediction + NIfTI export is now a mode of the unified eval
script. This thin shim preserves the old command line by translating it into:

    python scripts/eval_ldm_7tcdm3d.py --subject <id> --save_nii \
        --sample_seed <seed> ...

Prefer calling eval_ldm_7tcdm3d.py directly:

    python scripts/eval_ldm_7tcdm3d.py --ckpt runs/ldm_7tcdm3d/stage2_best.pt \
        --subject 2024_040 --save_nii --seed 0
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.eval_ldm_7tcdm3d import main as eval_main


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt",           required=True)
    p.add_argument("--subject",        required=True, help="Subject id, e.g. 2024_040")
    p.add_argument("--data_root",      default="fmri")
    p.add_argument("--out_dir",        default=None)
    p.add_argument("--ddim_steps",     type=int,   default=None)
    p.add_argument("--guidance_scale", type=float, default=None)
    p.add_argument("--eta",            type=float, default=0.0)
    p.add_argument("--seed",           type=int,   default=42)
    p.add_argument("--n_samples",      type=int,   default=1)
    args = p.parse_args()

    print("[predict_ldm_7tcdm3d_nii.py is deprecated — forwarding to "
          "eval_ldm_7tcdm3d.py --subject ... --save_nii]")

    argv = [
        "eval_ldm_7tcdm3d.py",
        "--ckpt",        args.ckpt,
        "--subject",     args.subject,
        "--data_root",   args.data_root,
        "--eta",         str(args.eta),
        "--sample_seed", str(args.seed),
        "--n_samples",   str(args.n_samples),
        "--save_nii",
    ]
    if args.out_dir is not None:
        argv += ["--out_dir", args.out_dir]
    if args.ddim_steps is not None:
        argv += ["--ddim_steps", str(args.ddim_steps)]
    if args.guidance_scale is not None:
        argv += ["--guidance_scale", str(args.guidance_scale)]

    sys.argv = argv
    eval_main()


if __name__ == "__main__":
    main()
