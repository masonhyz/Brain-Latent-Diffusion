#!/bin/bash
# Train the 7TCDM-3D latent diffusion model with 7-fold cross-validation.
#
# Each fold is a full, independent two-stage run (KL-autoencoder then latent
# denoiser) on that fold's 6/7 training subjects, validated on the held-out 1/7.
# Across the 7 folds every subject is validated exactly once, so aggregating the
# folds gives a split-robust estimate (mean ± std) instead of a single lucky
# 15% holdout.
#
# Run from the project root:
#   bash scripts/train_7tcdm3d.sh                 # -> runs/ldm_7tcdm3d_<ts>_fold{0..6}/
#   bash scripts/train_7tcdm3d.sh my_experiment   # -> runs/my_experiment_fold{0..6}/
#   bash scripts/train_7tcdm3d.sh my_experiment 3 # -> only fold 3 (resume / 1-GPU-per-fold)
set -e

# Base run name; each fold appends _fold<k>. Default: fresh timestamp so a new
# kickoff never overwrites an old sweep.
RUN="${1:-ldm_7tcdm3d_$(date +%Y-%m-%d_%H-%M-%S)}"
N_FOLDS=7
SEED=42

# Optional 2nd arg runs a single fold instead of all 7 — handy for resuming a
# crashed fold or fanning folds out across GPUs. Default: every fold, in order.
if [ -n "$2" ]; then
    FOLDS=("$2")
else
    FOLDS=($(seq 0 $((N_FOLDS - 1)))) # 0 1 2 3 4 5 6
fi

for FOLD in "${FOLDS[@]}"; do
    OUT="runs/${RUN}_fold${FOLD}"
    echo "========================================================================"
    echo " Fold ${FOLD} / ${N_FOLDS}   ->   ${OUT}"
    echo "========================================================================"

    # ── Stage 1: KL-autoencoder (this fold's train split) ─────────────────────
    python scripts/train_ldm_7tcdm3d.py \
        --stage 1 \
        --out_dir "${OUT}" \
        --data_root fmri \
        --epochs 200 \
        --n_folds ${N_FOLDS} --fold ${FOLD} \
        --seed ${SEED}

    # ── Stage 2: latent denoiser, SAME fold, AE frozen ────────────────────────
    # Same seed + n_folds + fold => identical held-out subjects as stage 1, and
    # references this fold's own stage-1 checkpoint (the AE is fold-specific).
    python scripts/train_ldm_7tcdm3d.py \
        --stage 2 \
        --ae_ckpt "${OUT}/stage1_best.pt" \
        --out_dir "${OUT}" \
        --data_root fmri \
        --epochs 2000 \
        --n_folds ${N_FOLDS} --fold ${FOLD} \
        --seed ${SEED}

    echo "Fold ${FOLD} done. Best checkpoint: ${OUT}/stage2_best.pt"
done

echo ""
echo "All requested folds complete. Aggregate the CV metrics with:"
echo "  python scripts/aggregate_kfold.py runs/${RUN}"
