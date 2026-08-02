#!/bin/bash
# Train the 7TCDM-3D latent diffusion model (KL-autoencoder then latent denoiser).
#
# Default is 7-fold cross-validation: each fold is a full, independent two-stage
# run on that fold's 6/7 training subjects, validated on the held-out 1/7. Across
# the 7 folds every subject is validated exactly once, so aggregating the folds
# gives a split-robust estimate (mean ± std) instead of a single lucky holdout.
#
# Run from the project root:
#   bash scripts/train_7tcdm3d.sh                     # 7-fold CV -> runs/<ts>_fold{0..6}/
#   bash scripts/train_7tcdm3d.sh my_experiment       # 7-fold CV -> runs/my_experiment_fold{0..6}/
#   bash scripts/train_7tcdm3d.sh my_experiment 3     # ONE fold 3 -> runs/my_experiment_fold3/
#   bash scripts/train_7tcdm3d.sh my_experiment holdout   # legacy single val_frac holdout
#
# The 2nd arg selects the mode:
#   (omitted)  -> all 7 folds, in order
#   <int>      -> only that fold (resume a crash / one GPU per fold)
#   holdout    -> the original single 15% holdout (200/35), NO k-fold
set -e

# Base run name; folds append _fold<k>. Default: fresh timestamp so a new kickoff
# never overwrites an old sweep.
RUN="${1:-ldm_7tcdm3d_$(date +%Y-%m-%d_%H-%M-%S)}"
MODE="$2"
N_FOLDS=7
SEED=42

# Run both stages into one output dir. $1 = out dir; any further args (the fold
# selection) are forwarded verbatim to BOTH stages, so stage 2 always shares
# stage 1's exact split and references this dir's own stage-1 AE.
run_two_stage() {
    local OUT="$1"; shift
    echo "========================================================================"
    echo " ${OUT}"
    echo "========================================================================"
    # ── Stage 1: KL-autoencoder ───────────────────────────────────────────────
    python scripts/train_ldm_7tcdm3d.py \
        --stage 1 \
        --out_dir "${OUT}" \
        --data_root fmri \
        --epochs 200 \
        --seed ${SEED} "$@"
    # ── Stage 2: latent denoiser, AE frozen ───────────────────────────────────
    python scripts/train_ldm_7tcdm3d.py \
        --stage 2 \
        --ae_ckpt "${OUT}/stage1_best.pt" \
        --out_dir "${OUT}" \
        --data_root fmri \
        --epochs 2000 \
        --seed ${SEED} "$@"
    echo "Done. Best checkpoint: ${OUT}/stage2_best.pt"
}

if [ "${MODE}" = "holdout" ]; then
    # Legacy behaviour: single random val_frac=0.15 holdout (200 train / 35 val),
    # no k-fold. Identical to the pre-CV pipeline. No --fold => holdout path.
    echo "Mode: legacy holdout (no k-fold)"
    run_two_stage "runs/${RUN}"
    echo ""
    echo "Holdout run complete. Eval with:"
    echo "  python scripts/eval_ldm_7tcdm3d.py --ckpt runs/${RUN}/stage2_best.pt --val_only"

elif [ -n "${MODE}" ]; then
    # Single fold of the ${N_FOLDS}-way split.
    echo "Mode: single fold ${MODE} of ${N_FOLDS}"
    run_two_stage "runs/${RUN}_fold${MODE}" --n_folds ${N_FOLDS} --fold ${MODE}
    echo ""
    echo "Fold ${MODE} complete. Aggregate available folds with:"
    echo "  python scripts/aggregate_kfold.py runs/${RUN}"

else
    # Full ${N_FOLDS}-fold cross-validation.
    echo "Mode: ${N_FOLDS}-fold cross-validation"
    for FOLD in $(seq 0 $((N_FOLDS - 1))); do
        run_two_stage "runs/${RUN}_fold${FOLD}" --n_folds ${N_FOLDS} --fold ${FOLD}
    done
    echo ""
    echo "All folds complete. Aggregate the CV metrics with:"
    echo "  python scripts/aggregate_kfold.py runs/${RUN}"
fi
