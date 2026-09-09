# Moyamoya Post-Surgery Whole-Brain fMRI Forecast with Generative Models

Predicting **post-surgery CBF** (cerebral blood flow) from **pre-surgery fMRI** in
Moyamoya patients, framed as **conditional 3D image-to-image generation**. Given a
pre-op CBF volume `x_pre`, the models generate the predicted 6-months-post-op volume
`x_post`.

The current focus is **Flow3D**, a conditional flow matching model. A family of
**3D diffusion models** precedes it, and older UNet / CVAE / CycleGAN baselines are
retained as [legacy models](#legacy-models).

## ⚠️ Three things to know before reading any number in this repo

### 1. The metrics are not brain-masked, and the mask is the wrong one

Every metric documented as "brain-masked" is in fact **whole-volume**, and the
brain is only ~29 % of that volume.

`ToChannelsFirstAndNormalize` computes z-score statistics over nonzero voxels but
applies the shift `(t - mean)/std` to the *whole* tensor, so background voxels —
exactly 0 on input — come out at `-mean/std ≈ -2.4`. Every downstream
`(x != 0)` mask therefore selects 100 % of the volume. Roughly 71 % of every
reported metric is measuring how well a model reproduces a constant background.

The mask is also a *union*, `(x_pre != 0) | (x_post != 0)`, which is wrong for a
comparison: ~5.4 % of it has tissue in one scan and nothing in the other, where no
pre-vs-post difference exists at all. On that sliver the two preprocessing
conventions disagree by 6× (mean |x−y| of 0.357 vs 2.034 from the same voxels);
over the *intersection* they agree exactly. Only the intersection is meaningful.

Corrected, on the seed=42 / `val_frac=0.15` holdout (n=35), the identity predictor
scores:

| Scoring | MAE ↓ | MSE ↓ | PSNR ↑ | SSIM ↑ |
|---------|-------|-------|--------|--------|
| As reported historically (whole volume) | 0.2231 | 0.1471 | 25.25 | 0.8382 |
| **Brain tissue only** (`--zero_background`, intersection mask) | **0.4717** | **0.4366** | **20.39** | **0.4269** |

The good news is in the last row: a real SSIM of 0.43 means there is far more
headroom on brain tissue than the whole-volume 0.84 suggested. Flow3D uses the
corrected convention (`--zero_background`, on by default, plus
`moyamoya.metrics.tissue_mask`). The diffusion models keep the old preprocessing so
their checkpoints still reproduce — **their numbers are not comparable to Flow3D's.**

### 2. The identity baseline beats every diffusion model here

Pre- and post-surgery volumes are the same brain six months apart, so simply
copying the input is a strong predictor. Scored the historical way, so that these
are like-for-like:

| Model | MAE ↓ | PSNR ↑ | SSIM ↑ |
|-------|-------|--------|--------|
| **Identity (copy `x_pre`)** | **0.2231** | **25.25** | **0.8382** |
| Best latent diffusion (`runs/2026-08-02_12-56-50_stage2`) | 0.2361 | 23.88 | 0.7997 |
| Best CDM3D (image space) | 0.3179 | 23.27 | 0.6387 |

(diffusion figures are each run's *best epoch*.)

Every diffusion model loses to doing nothing, on every metric. The cause is
structural rather than a tuning failure: a model that generates `x_post` starting
from Gaussian noise must re-synthesise the entire brain, when most of the answer
was already in the input — and 235 subjects is not enough to win that way. It is
not the autoencoder's fault either: encoding and decoding the *true* `x_post`
scores MAE 0.1494 / PSNR 31.63 / SSIM 0.9648, so the latent space is not the
binding constraint. The generative process is.

This is what [Flow3D](#flow3d--conditional-flow-matching) is designed to fix, and
why `moyamoya/metrics.py::identity_baseline` is computed and printed by every
Flow3D training and evaluation run.

### 3. Most of the pre→post "change" is registration noise, not surgery

This is the one that most threatens the paper's central claim, and the one the
first two warnings can hide behind. Identity wins because the volumes are nearly
identical — so the only place a model can add value is the small **region of
change** (`moyamoya/metrics.py::change_mask`, the top-5 % most-changed voxels). It
is tempting to headline "we cut error by X% in the region surgery altered." But
pre- and post-op are **separate acquisitions ~6 months apart, warped to a common
space**, and measured across all 235 pairs that change region is mostly *noise*,
not biology:

- **~60 % of the change energy in the ROI is high-frequency** (σ=2 residual); only
  ~40 % is coherent, low-frequency structure. `change_region_report(coherent=True)`
  reports this as `coherent_frac`.
- **The ROI is 2.2× enriched on `x_pre`'s structural edges** (0.448 vs 0.20 by
  chance) — the classic **mis-registration signature**: a sub-voxel warp error at a
  moving anatomical edge produces a large `|Δ|` that has nothing to do with
  perfusion. Reported as `edge_enrichment`.
- The ROI change is **50/50 in sign** and only weakly lateralised (0.54), *not* the
  coherent one-sided "revascularisation raises CBF" effect one would hope to model.
- A population mean-change template explains individuals at **R² ≈ 0.02** — there is
  no reusable fixed answer; localisation must be inferred per subject.
- An **oracle smooth edit** (`x_pre` + Gaussian-blurred true Δ) removes ~62 % of the
  ROI error — i.e. the *recoverable* signal is the low-frequency part.

**What this means for the paper.** Do not report raw `change_mae` /
`change_mae_improvement` as "predicting the surgical effect": a large part of that
ROI error is registration/acquisition noise that no model can or should predict, so
both the model's error and the identity bar are inflated by noise, and chasing it is
exactly what produced the blurry and hallucinated results in the change-aware and
adversarial experiments. Report instead the **noise-aware** numbers from the opt-in
**coherent** family (`change_region_report(coherent=True)`, or `--coherent` on the
train/eval scripts; off by default so the standard metrics are unchanged):

- `coherent_frac` — the fraction of the ROI that is recoverable signal (the rest is
  the **registration noise floor**); state it explicitly.
- `coherent_mae` / `coherent_mae_improvement` (and `coherent_{mse,psnr,ssim}`) —
  error and skill on the **coherent (low-frequency) edit** only, the part that is
  actually predictable.
- `edge_enrichment` — the mis-registration signature, as a caveat.

Frame the contribution as predicting the **coherent** post-op change, and name
mis-registration as a limitation: **improving pre/post co-registration would likely
raise the ceiling more than any change to the loss** — a data-side fix this repo has
flagged but not yet run.

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
  metrics.py               # brain-masked MAE / MSE / PSNR / SSIM + identity_baseline
  runlog.py                # shared run plumbing: console tee, hparams, CSV, W&B, plots
  utils.py                 # seeding, masked L1, union mask
  models/
    flow3d.py              # Flow3D — conditional flow matching (current focus)
    ldm.py                 # PairedDiffusion — image-space conditional diffusion (no VAE)
    cdm3d.py               # CDM3D — image-space x0-prediction diffusion (7TCDM-style UNet)
    ldm3d.py               # PairedLatentDiffusion — true 2-stage latent diffusion (KL-AE + UNet)
    ldm_7tcdm3d.py         # PairedLatentDiffusion with a 7TCDM-derived latent denoiser
    unet.py, cvae.py, cyclegan_wrapper.py   # legacy baselines
  viz/                     # HTML training visualizer + slice-grid plotting

scripts/                   # CLI entry points (run from project root)
tests/test_flow3d.py       # Flow3D self-checks (`python tests/test_flow3d.py`)
third_party/               # external repos (clone separately; not committed)
  latent-diffusion/        #   CompVis LDM — 3D-converted Encoder/Decoder used by ldm3d
  7TCDM/                   #   source architecture ported into cdm3d / ldm_7tcdm3d
  pytorch-CycleGAN-and-pix2pix/     # legacy
  3D-CycleGan-Pytorch-MedImaging/   # legacy
```

## Flow3D — conditional flow matching

`moyamoya/models/flow3d.py`. A flow model learns a velocity field and transports one
distribution to another by integrating an ODE. Unlike diffusion, **the source
distribution is a free choice — it does not have to be noise.** Flow3D uses that
freedom: by default it transports `x_pre → x_post` directly (a *bridge*, or
data-to-data coupling) instead of `noise → x_post`.

```
x_t = (1-t)·x_pre + t·x_post + γ(t)·z ,   z ~ N(0,I),  γ(t) = σ·sin(πt)
u_t = (x_post - x_pre) + γ'(t)·z                              ← regression target
loss = E ‖ v_θ(x_t, t | x_pre) - u_t ‖²

sampling: integrate dx/dt = v_θ(x_t, t | x_pre) from x(0) = x_pre to t = 1
```

Three properties follow, and they matter more than the architecture:

1. **Training starts at the identity baseline.** With `--zero_init_out` (default)
   the output conv is zero-initialised, so `v_θ ≡ 0` and an *untrained* model
   returns `x_pre` exactly. The model begins tied with the best thing this repo has
   produced and only has to learn the residual — instead of spending its first
   thousand epochs relearning brain anatomy.
2. **The path is short, so sampling is cheap.** `‖x_post − x_pre‖` is small (masked
   MAE 0.24 against a volume of masked std 1.10), the ODE is nearly straight, and a
   handful of steps suffices — 8 Heun steps is 16 NFE, against 50 for DDIM. Use
   `eval_flow3d.py --sweep_steps` to measure the accuracy/NFE curve on your own run.
3. **No ill-conditioned reparameterisation.** DDIM's x₀↔ε conversion divides by
   √(1−ᾱ) ≈ 0.01 at late steps, which is why `cdm3d.sample` must force fp32 and
   clamp every step. The velocity is regressed and integrated directly; nothing
   divides by a vanishing quantity.

`γ(t) = σ·sin(πt)` rather than the Brownian-bridge `σ√(t(1−t))`, whose derivative
diverges at both endpoints and would give the regression target unbounded variance
exactly where it matters. σ is a **smoothing** knob, not a diversity knob: the
probability-flow ODE is deterministic and γ(0)=0, so one `x_pre` gives one
prediction whatever σ is. What σ>0 buys is a velocity field trained on a
neighbourhood of the path, which makes integration error self-correcting. For an
ensemble, perturb the start with `--init_noise`.

`--source noise` recovers standard CFM from a Gaussian (rectified flow). It is kept
so the objective can be compared against the diffusion models with the *architecture
held fixed*: the velocity net is `Unet3D_CDM`, the exact denoiser CDM3D uses, so a
CDM3D-vs-Flow3D comparison isolates the objective and the sampler.

### The conditioning trap

**The bridge must not receive `x_pre` as a network input**, even though every
diffusion model in this repo conditions exactly that way. Along the training path
`x_t = (1-t)·x_pre + t·x_post`, so a network handed both `x_t` and `x_pre` can
simply return

```
x_post - x_pre  =  (x_t - x_pre) / t
```

That is an algebraic identity. It drives the training loss to ~0 while learning
*nothing* about how pre-op perfusion maps to post-op perfusion, and it is what the
network will find, because it is far easier than the real task. Measured on pairs
whose residual is pure unpredictable noise — honest loss floor 0.25 — the concat
model reaches **0.003**, 80× below the floor.

Sampling then collapses: with the shortcut the trajectory from `x(0) = x_pre` is
`x_pre + t·v₀`, so the entire prediction is decided by the velocity at `t=0`, the
one slice of the path the shortcut leaves untrained. This shows up in training as a
healthy-looking loss curve alongside sampled metrics that drift *away* from the
identity baseline.

So `--condition` defaults to `none` for `--source pre`: the network sees `x_t` and
`t` only. Nothing is lost — `x_t` *is* `x_pre` at `t=0` and carries the subject's
anatomy at every `t`. `--source noise` keeps `concat`, where conditioning is
indispensable because `x_t` starts as pure noise. `tests/test_flow3d.py::
test_no_algebraic_shortcut` is the regression test.

### Training

```bash
# default: bridge from the pre-op volume
GPU=0 python scripts/train_flow3d.py --out_dir runs/flow3d_bridge --epochs 500

# ablation: standard CFM from noise, same U-Net, CFG as in the diffusion models
GPU=0 python scripts/train_flow3d.py --source noise --sigma 0 \
    --cfg_drop_prob 0.15 --guidance_scale 3.0 --out_dir runs/flow3d_noise
```

Flow flags: `--source {pre,noise}`, `--sigma`, `--t_dist {uniform,logit_normal}`
(the latter is the SD3 schedule), `--l1_weight`, `--ssim_weight`.
Solver flags: `--steps`, `--solver {euler,heun,rk4}`, `--val_steps`, `--init_noise`.
Net flags: `--dim`, `--dim_mults`, `--init_kernel_size` (7; drop to 3 under 12 GB),
`--zero_init_out/--no-zero-init-out`.
Optimisation: cosine LR with warmup, EMA (`--ema_beta`, `--ema_warmup`), grad
clipping, bf16 autocast. Validation metrics are always sampled from the EMA weights,
which is what eval loads.

Every epoch's metrics are printed against the identity baseline with a `[vs identity:
+++-]` flag per metric, and the run ends with an explicit per-metric verdict. Set
`GPU=<n>` to choose a device — `moyamoya/utils.py::get_device` defaults to GPU **1**,
and the training script prints free VRAM via `torch.cuda.mem_get_info` (nvidia-smi is
unreliable on this host, and a full GPU surfaces as a cryptic NVML assert, not an OOM).

### Evaluation

```bash
# validation split, model vs identity, per-subject win rate
python scripts/eval_flow3d.py --ckpt runs/flow3d_bridge/best_mae.pt --val_only

# accuracy vs NFE — how few ODE steps you can afford
python scripts/eval_flow3d.py --ckpt runs/flow3d_bridge/best_mae.pt --val_only \
    --sweep_steps 1 2 4 8 16 32

# one subject, 5-trajectory ensemble, export NIfTI in the source geometry
python scripts/eval_flow3d.py --ckpt runs/flow3d_bridge/best_mae.pt \
    --subject 2024_040 --n_samples 5 --init_noise 0.05 --save_nii
```

`metrics.csv` carries both the model's and the identity baseline's per-subject
scores, and `summary.json` records a `beats_identity` verdict plus the per-subject
MAE win rate — an aggregate can hide a model that helps a few subjects and harms
the rest.

### Self-checks

```bash
python tests/test_flow3d.py     # 8 checks, no pytest required
```

The load-bearing one is the oracle test: if the network returned the *true*
conditional velocity, every solver must land on `x_post` exactly, at any step count.
A sign error or a mis-scaled time in the interpolant, the target, or any solver
breaks it. The suite also checks the γ boundary conditions, that zero-init
reproduces `x_pre` bit-exactly, that the background stays exactly zero, and that the
objective can overfit a single pair (it reduces MAE by 99 %).

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

The external repos are not vendored; clone them into `third_party/`. See
[third_party/README.md](third_party/README.md) for the full list, exact paths, and
which model uses each.

```bash
mkdir -p third_party
git clone https://github.com/CompVis/latent-diffusion third_party/latent-diffusion
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

End-to-end (both stages, one shared run dir):

```bash
bash scripts/train_7tcdm3d.sh                 # -> runs/ldm_7tcdm3d_<timestamp>/
bash scripts/train_7tcdm3d.sh my_experiment   # -> runs/my_experiment/
```

Or run the stages directly. With **no `--out_dir`, stage 1 creates a fresh
timestamped dir** (`runs/ldm_7tcdm3d_<YYYY-MM-DD_HH-MM-SS>/`) so a new kickoff
never overwrites an old run; **stage 2 with no `--out_dir` defaults to the AE
checkpoint's own directory**, so it always lands next to the stage 1 it builds on:

```bash
# Stage 1 — prints the timestamped dir it created
python scripts/train_ldm_7tcdm3d.py --stage 1

# Stage 2 — lands in runs/<that dir>/ automatically (same dir as the AE)
python scripts/train_ldm_7tcdm3d.py --stage 2 \
    --ae_ckpt runs/ldm_7tcdm3d_<timestamp>/stage1_best.pt --epochs 2000
```

Pass `--out_dir runs/<name>` to either stage to pin a specific directory.
Denoiser flags: `--diff_dim`, `--diff_dim_mults`, `--init_kernel_size`, `--resnet_groups`.
A pinned-dir reproduction (both stages, seed 42) is scripted:

```bash
bash scripts/repro_ldm_7tcdm3d.sh
```

## Inference & evaluation

### Evaluate a single model

```bash
# CDM3D — validation split + full dataset, EMA weights, brain-masked metrics
python scripts/eval_cdm3d.py --ckpt runs/cdm3d/best.pt --ddim_steps 50

# 7TCDM-3D latent diffusion — writes to outputs/eval/ldm_7tcdm3d/
python scripts/eval_ldm_7tcdm3d.py                 # full dataset
python scripts/eval_ldm_7tcdm3d.py --val_only      # validation split only
python scripts/eval_ldm_7tcdm3d.py --no-grids      # metrics only, skip PNGs (fast)
```

All four metrics (MAE, MSE, PSNR, SSIM) are computed by one shared function
(`moyamoya/metrics.py`) so training, eval, and every comparison script report the
same numbers. They are scored in z-score space and **brain-masked** — including
SSIM, whose per-voxel SSIM map is averaged over the brain mask (set
`mask_ssim=False` for legacy whole-volume SSIM). PSNR and SSIM share one
`data_range` (the masked target's max−min; pass a fixed value after rescaling to
[0,1] for cross-study reporting). The
7TCDM-3D eval writes `grids/<id>.png` (per-sample 3×3 grids, toggle with
`--save_grids/--no-grids`), `metrics.csv` (per-sample) and `summary.json` (run config
+ aggregate mean/std/median/min/max). Sampling is seeded per subject
(`--sample_seed`), so results are reproducible and match single-subject runs.

### Evaluate all models together

```bash
python scripts/eval_all_models.py
```

Runs UNet, CycleGAN, LDM-3D and 7TCDM-3D on the full paired dataset and reports a
unified metrics table. (Edit the checkpoint paths at the top of the script.)

### Predict one subject + export NIfTI

Single-subject prediction is a mode of the same eval script (`--subject`, plus
`--save_nii` to write the full predicted volume back to `.nii.gz` in the original
geometry). Defaults to `outputs/predict/<subject>/`; use `--n_samples N` to average an
ensemble of DDIM draws:

```bash
python scripts/eval_ldm_7tcdm3d.py --ckpt runs/ldm_7tcdm3d/stage2_best.pt \
    --subject 2024_040 --save_nii --sample_seed 0
```

> `scripts/predict_ldm_7tcdm3d_nii.py` still works as a thin deprecated shim that
> forwards to the command above.

### Comparison plots & figures

`scripts/` also contains plotting utilities for reports/posters:
`compare_models_boxplot.py`, `plot_compare_boxplots.py`, `compare_metrics.py`,
`plot_training_progression.py`, `plot_sample_poster.py`, `viz_cdm3d_full.py`, `eda.py`.

## Checkpoints

```
runs/<name>/stage1_best.pt   # LDM autoencoder (2-stage models) — the only copy of the AE
runs/<name>/stage2_best.pt   # LDM diffusion UNet (denoiser); references the AE, not embedded
runs/<name>/best.pt          # single-stage models (CDM3D, PairedDiffusion) — best val
runs/<name>/last.pt          # latest epoch
```

Checkpoints embed their build `args`, so eval/predict scripts reconstruct the exact model
variant from the checkpoint alone. The AE is frozen during stage 2, so stage-2 checkpoints
store **only the denoiser** plus an `ae_ckpt` reference — they do not duplicate the ~160 MB
AE. Loading is handled by `moyamoya.models.ldm_7tcdm3d.load_7tcdm3d_checkpoint`, which
resolves the AE from `stage1_best.pt` (or `--ae_ckpt`), and still reads legacy checkpoints
that embed the AE. **Keep `stage1_best.pt` alongside the stage-2 files** — it is required to
load them.

## Outputs & logging

Training and evaluation write to two trees. `runs/<name>/` holds everything tied to a
training run; `outputs/` holds evaluation, prediction, and figure artifacts, namespaced
by kind so a run's products are self-contained and easy to find.

```
runs/<name>/                    # one training run (7TCDM-3D shown)
  hparams_stage{1,2}.json       # args + dataset split + timestamp + git commit
  metrics_stage{1,2}.csv        # per-epoch scalars (the durable history)
  train_stage{1,2}.log          # full console output, teed per run (append)
  stage1_{best,last}.pt         # autoencoder (the sole AE copy)
  stage2_{best,last}.pt         # denoiser only (+ ae_ckpt reference; AE not duplicated)
  stage2_best_{mae,mse,psnr,ssim}.pt   # per-metric best checkpoints (denoiser only)
  vis_stage2/epoch_XXXX.png     # sampled val grid every --vis_every epochs
  training_progression*.png     # summary plot, rebuilt from the CSVs

outputs/                        # both trees written by scripts/eval_ldm_7tcdm3d.py
  eval/<run>/                   #   full-dataset / --val_only runs
    grids/<id>.png              #     per-sample 3×3 grids (--save_grids)
    metrics.csv                 #     per-sample metrics
    summary.json                #     run config + aggregate stats
  predict/<subject>/            #   --subject runs
    grids/<subject>.png, <subject>_pred.nii.gz (--save_nii), metrics.csv, summary.json
```

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
