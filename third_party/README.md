# third_party

External repositories this project depends on. This folder is **git-ignored**
(except this README) — the upstream code is **not vendored** into our repo, so you
have to clone each dependency locally into the paths below before the models that
use them will import.

## Layout

```
third_party/
  latent-diffusion/                  # CompVis LDM — 3D-converted Encoder/Decoder
  7TCDM/                             # source architecture ported into our models
  pytorch-CycleGAN-and-pix2pix/      # legacy 2D CycleGAN / pix2pix backend
  3D-CycleGan-Pytorch-MedImaging/    # legacy 3D CycleGAN backend
  README.md                          # this file (tracked)
```

## Dependencies

| Directory | Upstream | Used by | Notes |
|-----------|----------|---------|-------|
| `latent-diffusion/` | [CompVis/latent-diffusion](https://github.com/CompVis/latent-diffusion) | `moyamoya/models/ldm3d.py`, `ldm_7tcdm3d.py` (LDM-3D, 7TCDM-3D) | Imports `Encoder`/`Decoder` from `ldm/modules/diffusionmodules/model.py`; expects the **3D-converted** version (Conv2d → Conv3d, etc.). Path is resolved at import time in `ldm3d.py` (`_LDM_ROOT`). |
| `7TCDM/` | 7TCDM source architecture | `moyamoya/models/cdm3d.py`, `ldm_7tcdm3d.py` | Reference source only — the UNet was **ported** into our code (see `cdm3d.py`, "3D port of 7TCDM's Unet3D"), not imported from here. Kept for provenance. |
| `pytorch-CycleGAN-and-pix2pix/` | [junyanz/pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) | `scripts/train_cyclegan_2d.py`, `infer_cyclegan_2d.py` | **Legacy** baseline. Delegated to as a subprocess/library. |
| `3D-CycleGan-Pytorch-MedImaging/` | [arnab39/cycleGAN-PyTorch](https://github.com/arnab39/cycleGAN-PyTorch) | `scripts/train_cyclegan_3d.py`, `infer_cyclegan_3d.py` | **Legacy** baseline. Cloned into this directory name. |

## Clone

```bash
mkdir -p third_party

# Required for the latent-diffusion models (LDM-3D, 7TCDM-3D):
git clone https://github.com/CompVis/latent-diffusion third_party/latent-diffusion
# NOTE: ldm3d.py needs the 3D-converted Encoder/Decoder (Conv2d -> Conv3d, etc.).

# Legacy CycleGAN baselines (only needed for the CycleGAN scripts):
git clone https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix third_party/pytorch-CycleGAN-and-pix2pix
git clone https://github.com/arnab39/cycleGAN-PyTorch third_party/3D-CycleGan-Pytorch-MedImaging
```

`7TCDM/` is a source reference that was ported into `moyamoya/models/`; obtain it
from its original source if you need to compare against the upstream architecture.
It is not required at runtime.

## Rules

- Keep each repo at exactly the path in the table above — import paths are hard-coded
  (e.g. `_LDM_ROOT` in `ldm3d.py`).
- Do not commit the cloned code; only this README is tracked. Everything else under
  `third_party/` stays ignored.
