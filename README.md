# Moyamoya fMRI Pre/Post Surgery Translation

Models for predicting post-surgery CBF (cerebral blood flow) from pre-surgery fMRI using paired and unpaired image-to-image translation.

## Project Structure

```
moyamoya/          # Python package (shared library code)
  config.py        # UNet training config dataclass
  dataset.py       # PrePostFMRI paired dataset loader
  transform.py     # NIfTI → PyTorch tensor, z-score normalization
  utils.py         # Seeding, masked L1, union mask
  modules.py       # Shared building blocks: DoubleConv3d, _match_size, DownBlock, UpBlock
  models/
    unet.py        # 3D UNet
    cvae.py        # 3D Conditional VAE
    cyclegan_wrapper.py  # CycleGAN dict-format dataset wrapper
  viz/
    html_viz.py    # Auto-refreshing HTML visualizer for 3D training

scripts/           # CLI entry points
  setup_data.py              # Copy NIfTI files from source into fmri/
  train_unet.py              # Train 3D UNet (paired, supervised)
  train_cvae.py              # Train 3D CVAE (paired, supervised)
  train_cyclegan_2d.py       # Train 2D pix2pix or CycleGAN on axial slices
  train_cyclegan_3d.py       # Train 3D CycleGAN (unpaired, masked cycle loss)
  infer_unet.py              # Single-sample UNet inference + visualization
  infer_cyclegan_2d.py       # Batch 2D CycleGAN inference
  infer_cyclegan_3d.py       # Batch 3D CycleGAN inference
  qc.py                      # Save orthogonal-slice QC plots for all pairs
  viz_check.py               # Quick 3×3 grid check for 3D CycleGAN outputs

third_party/       # External repos (not committed — clone separately)
  pytorch-CycleGAN-and-pix2pix/
  3D-CycleGan-Pytorch-MedImaging/
```

## Setup

### 1. Clone third-party dependencies

The scripts depend on two external repos. Clone them into `third_party/`:

```bash
mkdir -p third_party
git clone https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix third_party/pytorch-CycleGAN-and-pix2pix
git clone https://github.com/arnab39/cycleGAN-PyTorch third_party/3D-CycleGan-Pytorch-MedImaging
```

### 2. Install dependencies

```bash
pip install torch torchvision nibabel numpy matplotlib
```

### 3. Prepare data

Data should live in `fmri/` at the project root:

```
fmri/
  pre_surgery/              # pre-op CBF volumes: <year>_<id>.nii.gz
  6_months_post_surgery/    # post-op CBF volumes: <year>_<id>.nii.gz
```

To copy files from the raw data source (edit the paths in the script first):

```bash
python scripts/setup_data.py
```

## Training

All scripts are run from the **project root**.

### 3D UNet (paired, supervised)

Predicts post-surgery CBF from pre-surgery CBF using masked L1 loss.

```bash
python scripts/train_unet.py
```

Config lives in `moyamoya/config.py` (edit `data_root`, `out_dir`, `epochs`, `base_channels`, etc.).

### 3D CVAE (paired, supervised)

Same as UNet but with a VAE bottleneck and KL loss.

```bash
python scripts/train_cvae.py
```

### 2D pix2pix / CycleGAN (axial slices)

Delegates to `pytorch-CycleGAN-and-pix2pix`. Recommended mode is `pix2pix` (paired).

```bash
# Paired pix2pix (recommended)
python scripts/train_cyclegan_2d.py --model pix2pix --name fmri_pix2pix

# Unpaired CycleGAN
python scripts/train_cyclegan_2d.py --model cycle_gan --name fmri_cyclegan_2d
```

### 3D CycleGAN (unpaired, masked cycle loss)

Trains a 3D CycleGAN using `3D-CycleGan-Pytorch-MedImaging`. Cycle/identity losses are restricted to brain-masked voxels.

```bash
python scripts/train_cyclegan_3d.py --dataroot fmri/ --name fmri_3d_cyclegan

# Reduce ngf if OOM
python scripts/train_cyclegan_3d.py --dataroot fmri/ --name fmri_3d_cyclegan --ngf 32
```

Key flags:
- `--ngf` — generator base filters (default 48; try 32 or 16 if OOM)
- `--n_epochs` — epochs at fixed LR (default 100)
- `--n_epochs_decay` — epochs with linear LR decay (default 100)
- `--norm_mode` — `zscore` (default) or `minmax`
- `--checkpoints_dir` — where to save checkpoints (default `checkpoints/`)

## Inference

### UNet — single sample

```bash
python scripts/infer_unet.py \
  --data_root fmri/ \
  --ckpt runs/unet_prepost_bc128/best.pt \
  --base_channels 128 \
  --sample_idx 0 \
  --save_fig outputs/infer.png
```

### 2D CycleGAN — batch

```bash
python scripts/infer_cyclegan_2d.py --name fmri_pix2pix
python scripts/infer_cyclegan_2d.py --name fmri_pix2pix --epoch 100 --max_subjects 10
```

### 3D CycleGAN — batch

```bash
python scripts/infer_cyclegan_3d.py --name fmri_3d_cyclegan
python scripts/infer_cyclegan_3d.py --name fmri_3d_cyclegan --which_epoch 50
```

## Quality Control

Save orthogonal-slice comparison plots (pre vs. post GT) for every subject:

```bash
python scripts/qc.py --data_root fmri/ --out_dir qc/
```

Quick visual check of 3D CycleGAN outputs for N subjects:

```bash
python scripts/viz_check.py --checkpoint checkpoints/fmri_3d_cyclegan --dataroot fmri/ --n 5
```

## Checkpoints

Checkpoints are saved to `checkpoints/<name>/` (CycleGAN) or `runs/<name>/` (UNet/CVAE):

```
runs/<name>/best.pt    # best validation loss
runs/<name>/last.pt    # latest epoch
checkpoints/<name>/    # CycleGAN epoch checkpoints
```
