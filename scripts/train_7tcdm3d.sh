#!/bin/bash
# Train the 7TCDM-3D latent diffusion model end-to-end (both stages).
# Run from the project root:   bash scripts/train_7tcdm3d.sh [run_name]
set -e

RUN="${1:-ldm_7tcdm3d}"          # run name -> runs/<RUN>/
OUT="runs/${RUN}"
SEED=42

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
