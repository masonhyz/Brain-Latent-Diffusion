# Moyamoya Post-Surgery Whole-Brain fMRI Forecast with Diffusion Models

Predicting **post-surgery CBF** (cerebral blood flow) from **pre-surgery fMRI** in
Moyamoya patients, framed as **conditional 3D image-to-image generation**. Given a
pre-op CBF volume `x_pre`, the models generate the predicted 6-months-post-op volume
`x_post`.

The current focus is a family of **3D diffusion models**. Older UNet / CVAE / CycleGAN
baselines are retained as [legacy models](#legacy-models).

## Data

Paired NIfTI volumes live in `fmri/` at the project root:

```
fmri/
  pre_surgery/              # pre-op CBF volumes:  <year>_<id>.nii.gz
  6_months_post_surgery/    # post-op CBF volumes: <year>_<id>.nii.gz
```

Volumes are loaded and z-score normalized (`moyamoya/transform.py`); a brain mask
(`voxel != 0`) is used to restrict losses and metrics to brain tissue. Full-resolution
volumes are `(91, 109, 91)`. The train/val split is deterministic given `(val_frac, seed)`
via `moyamoya/data.py::reconstruct_val_split`, so a checkpoint's held-out validation
subjects are exactly reproducible at eval time.

## Project structure

```
moyamoya/                  # Python package (shared library)
  data.py                  # deterministic train/val split + dataloaders
  dataset.py               # PrePostFMRI paired dataset
  transform.py             # NIfTI → tensor, z-score normalization, paired augmentation
  modules.py               # shared 3D blocks: ResBlock3D, ResnetBlock3D, Down/Upsample3D,
                           #   sinusoidal_embedding, make_beta_schedule, _match_size, ...
  metrics.py               # brain-masked MAE / MSE / PSNR / SSIM
  utils.py                 # seeding, masked L1, union mask
  models/
    ldm.py                 # PairedDiffusion — image-space conditional diffusion (no VAE)
    cdm3d.py               # CDM3D — image-space x0-prediction diffusion (7TCDM-style UNet)
    ldm3d.py               # PairedLatentDiffusion — true 2-stage latent diffusion (KL-AE + UNet)
    ldm_7tcdm3d.py         # PairedLatentDiffusion with a 7TCDM-derived latent denoiser
    unet.py, cvae.py, cyclegan_wrapper.py   # legacy baselines
  viz/                     # HTML training visualizer + slice-grid plotting

scripts/                   # CLI entry points (run from project root)
third_party/               # external repos (clone separately; not committed)
  latent-diffusion/        #   CompVis LDM — 3D-converted Encoder/Decoder used by ldm3d
  7TCDM/                   #   source architecture ported into cdm3d / ldm_7tcdm3d
  pytorch-CycleGAN-and-pix2pix/     # legacy
  3D-CycleGan-Pytorch-MedImaging/   # legacy
```

## The diffusion models

All four are conditioned on `x_pre` by channel concatenation and trained with
classifier-free guidance (CFG) dropout, so guidance scale can be tuned at sampling
time. Sampling uses DDIM (`eta=0` → deterministic).

| Model | File / builder | Space | Denoiser | Parameterization | Stages |
|-------|----------------|-------|----------|------------------|--------|
| **PairedDiffusion** | `ldm.py` · `build_paired_diffusion` | Image | ResNet 3D UNet | ε-prediction | 1 |
| **CDM3D** | `cdm3d.py` · `build_cdm3d` | Image | 7TCDM depthwise UNet | x₀-prediction + SSIM loss, EMA | 1 |
| **LDM-3D** | `ldm3d.py` · `build_paired_latent_diffusion` | Latent | 3D UNet | ε-prediction | 2 (KL-AE → diffusion) |
| **7TCDM-3D** | `ldm_7tcdm3d.py` · `build_paired_latent_diffusion_7tcdm` | Latent | 7TCDM ResNet UNet | ε-prediction | 2 (KL-AE → diffusion) |

- **PairedDiffusion** (`ldm.py`) — the simplest: diffusion runs directly in image space,
  `input = cat([x_t, x_pre])`. With `eta=0` it behaves like deterministic regression.
- **CDM3D** (`cdm3d.py`) — single-stage image-space model ported from 7TCDM. Predicts the
  clean image (x₀) rather than noise, uses a continuous cosine timestep schedule, a
  `MSE + λ·(1−SSIM3D)` loss, depthwise-separable convs, and EMA weights for inference.
- **LDM-3D** (`ldm3d.py`) — a true two-stage latent diffusion model: a KL-regularized 3D
  autoencoder (CompVis Encoder/Decoder, 3D-converted) is trained first, then a latent-space
  diffusion UNet is trained with the AE frozen.
- **7TCDM-3D** (`ldm_7tcdm3d.py`) — same two-stage LDM wrapper and AE as LDM-3D, but the
  latent denoiser is a 3D port of 7TCDM's UNet (more ResBlocks per level, scale-shift time
  conditioning).

## Setup

### 1. Clone third-party dependencies

```bash
mkdir -p third_party
git clone https://github.com/CompVis/latent-diffusion third_party/latent-diffusion
# 7TCDM source architecture (ported into cdm3d / ldm_7tcdm3d)
# Legacy CycleGAN backends:
git clone https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix third_party/pytorch-CycleGAN-and-pix2pix
git clone https://github.com/arnab39/cycleGAN-PyTorch third_party/3D-CycleGan-Pytorch-MedImaging
```

> Note: `ldm3d.py` imports the CompVis `Encoder`/`Decoder` from
> `third_party/latent-diffusion` and expects the **3D-converted** version (Conv2d →
> Conv3d, etc.).

### 2. Install dependencies

```bash
pip install torch torchvision nibabel numpy matplotlib einops
```

### 3. Prepare data

Place data in `fmri/` as shown [above](#data), or copy from the raw source
(edit the paths in the script first):

```bash
python fmri/setup_data.py
```

## Training

Run everything from the **project root**. Common flags: `--data_root`, `--out_dir`,
`--epochs`, `--batch_size`, `--lr`, `--amp/--no-amp`, `--val_frac`, `--seed`,
`--ddim_steps`, `--cfg_drop_prob`, `--guidance_scale`, `--vis_every`.

### PairedDiffusion (image-space, 1 stage)

```bash
python scripts/train_ldm.py --out_dir runs/paired_diff
```

### CDM3D (image-space x₀-prediction, 1 stage)

```bash
python scripts/train_cdm3d.py --data_root fmri --out_dir runs/cdm3d
```

Model flags: `--dim`, `--dim_mults`, `--init_kernel_size` (reduce to 3 on <12 GB GPUs),
`--ssim_weight`, `--ema_beta`, `--T`. EMA weights are saved alongside the online model.

### LDM-3D (latent diffusion, 2 stages)

```bash
# Stage 1 — train the KL autoencoder
python scripts/train_ldm3d.py --stage 1 --out_dir runs/ldm3d

# Stage 2 — freeze the AE, train the latent diffusion UNet
python scripts/train_ldm3d.py --stage 2 \
    --ae_ckpt runs/ldm3d/stage1_best.pt \
    --out_dir runs/ldm3d
```

AE flags: `--z_channels`, `--embed_dim`, `--ae_ch`, `--ae_res_blocks`, `--kl_weight`.
Diffusion flags: `--diff_base`, `--t_dim`, `--n_levels`.

### 7TCDM-3D (latent diffusion, 2 stages)

```bash
# Stage 1
python scripts/train_ldm_7tcdm3d.py --stage 1 --out_dir runs/ldm_7tcdm3d

# Stage 2
python scripts/train_ldm_7tcdm3d.py --stage 2 \
    --ae_ckpt runs/ldm_7tcdm3d/stage1_best.pt \
    --out_dir runs/ldm_7tcdm3d --epochs 200
```

Denoiser flags: `--diff_dim`, `--diff_dim_mults`, `--init_kernel_size`, `--resnet_groups`.
A full reproduction (both stages, seed 42) is scripted:

```bash
bash scripts/repro_ldm_7tcdm3d.sh
```

## Inference & evaluation

### Evaluate a single model

```bash
# CDM3D — validation split + full dataset, EMA weights, brain-masked metrics
python scripts/eval_cdm3d.py --ckpt runs/cdm3d/best.pt --ddim_steps 50

# 7TCDM-3D latent diffusion — per-sample 3×3 grids + metrics.csv
python scripts/eval_ldm_7tcdm3d.py
```

Metrics (MAE, MSE, PSNR, SSIM) are computed in z-score space, brain-masked.

### Evaluate all models together

```bash
python scripts/eval_all_models.py
```

Runs UNet, CycleGAN, LDM-3D and 7TCDM-3D on the full paired dataset and reports a
unified metrics table. (Edit the checkpoint paths at the top of the script.)

### Export a prediction as NIfTI

Generate a prediction for one subject and write the full predicted volume back to
`.nii.gz` in the original geometry (source affine/header), plus a 3×3 PNG grid:

```bash
python scripts/predict_ldm_7tcdm3d_nii.py --seed 0
```

### Comparison plots & figures

`scripts/` also contains plotting utilities for reports/posters:
`compare_models_boxplot.py`, `plot_compare_boxplots.py`, `compare_metrics.py`,
`plot_training_progression.py`, `plot_sample_poster.py`, `viz_cdm3d_full.py`, `eda.py`.

## Checkpoints

```
runs/<name>/stage1_best.pt   # LDM autoencoder (2-stage models)
runs/<name>/stage2_best.pt   # LDM diffusion UNet
runs/<name>/best.pt          # single-stage models (CDM3D, PairedDiffusion) — best val
runs/<name>/last.pt          # latest epoch
```

Checkpoints embed their build `args`, so eval/predict scripts reconstruct the exact model
variant from the checkpoint alone.

---

## Legacy models

These pre-diffusion baselines still work but are no longer the focus.

### 3D UNet (paired, supervised)

Predicts post-surgery CBF with masked L1 loss. Config in `moyamoya/config.py`.

```bash
python scripts/train_unet.py
python scripts/infer_unet.py --data_root fmri/ --ckpt runs/unet_prepost_bc128/best.pt \
    --base_channels 128 --sample_idx 0 --save_fig outputs/infer.png
```

### 3D CVAE (paired, supervised)

UNet with a VAE bottleneck and KL loss.

```bash
python scripts/train_cvae.py
```

### 2D pix2pix / CycleGAN (axial slices)

Delegates to `third_party/pytorch-CycleGAN-and-pix2pix`.

```bash
python scripts/train_cyclegan_2d.py --model pix2pix --name fmri_pix2pix     # paired (recommended)
python scripts/train_cyclegan_2d.py --model cycle_gan --name fmri_cyclegan_2d
python scripts/infer_cyclegan_2d.py --name fmri_pix2pix
```

### 3D CycleGAN (unpaired, masked cycle loss)

Delegates to `third_party/3D-CycleGan-Pytorch-MedImaging`; cycle/identity losses are
restricted to brain-masked voxels.

```bash
python scripts/train_cyclegan_3d.py --dataroot fmri/ --name fmri_3d_cyclegan   # add --ngf 32 if OOM
python scripts/infer_cyclegan_3d.py --name fmri_3d_cyclegan
```

### Quality control

```bash
python fmri/qc.py    # orthogonal-slice pre-vs-post grids -> fmri/qc/ (see fmri/README.md)
```
