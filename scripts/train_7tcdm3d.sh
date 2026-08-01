#!/bin/bash
# Train the 7TCDM-3D latent diffusion model end-to-end (both stages).
# Run from the project root:
#   bash scripts/train_7tcdm3d.sh              # -> runs/ldm_7tcdm3d_<timestamp>/
#   bash scripts/train_7tcdm3d.sh my_experiment   # -> runs/my_experiment/
set -e

# Default to a fresh timestamped run dir so a new kickoff never overwrites an
# old one. Both stages share this dir. Pass a name to override.
RUN="${1:-ldm_7tcdm3d_$(date +%Y-%m-%d_%H-%M-%S)}"
OUT="runs/${RUN}"
SEED=42
echo "Run dir: ${OUT}"

# ── Stage 1: KL-autoencoder ───────────────────────────────────────────────────
python scripts/train_ldm_7tcdm3d.py \
    --stage 1 \
    --out_dir "${OUT}" \
    --data_root fmri \
    --epochs 200 \
    --seed ${SEED}

# ── Stage 2: latent diffusion denoiser (AE frozen) ────────────────────────────
# Uses the stage-1 checkpoint written above; keep it in ${OUT} — stage-2
# checkpoints reference it rather than embedding the AE.
python scripts/train_ldm_7tcdm3d.py \
    --stage 2 \
    --ae_ckpt "${OUT}/stage1_best.pt" \
    --out_dir "${OUT}" \
    --data_root fmri \
    --epochs 2000 \
    --seed ${SEED}

echo "Done. Best checkpoint: ${OUT}/stage2_best.pt"
