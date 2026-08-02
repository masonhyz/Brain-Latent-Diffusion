#!/bin/bash
# Train the 7TCDM-3D latent diffusion model (KL-autoencoder then latent denoiser).
#
# The 2nd arg selects the mode:
#   (omitted)  -> full k-fold CV: every fold trains BOTH stages on its 6/7 split
#   <int>      -> only that fold, both stages (resume a crash / one GPU per fold)
#   holdout    -> the original single 15% holdout (200/35), NO k-fold
#   cv2        -> train the AE ONCE (holdout), then k-fold CV on stage 2 only
#
# `cv2` is the recommended default: the AE shows no train/val generalization gap
# (val recon loss ~= train), so reusing one frozen AE across the stage-2 folds
# adds negligible leakage while skipping 6 redundant AE trainings. The thing we
# actually cross-validate — post-surgery prediction — is the stage-2 denoiser.
#
# Run from the project root:
#   bash scripts/train_7tcdm3d.sh my_experiment          # full CV, both stages
#   bash scripts/train_7tcdm3d.sh my_experiment cv2      # AE once + CV stage 2
#   bash scripts/train_7tcdm3d.sh my_experiment 3        # only fold 3, both stages
#   bash scripts/train_7tcdm3d.sh my_experiment holdout  # legacy single holdout
set -e

RUN="${1:-ldm_7tcdm3d_$(date +%Y-%m-%d_%H-%M-%S)}"
MODE="$2"
N_FOLDS=7
SEED=42

# ── stage runners ─────────────────────────────────────────────────────────────
# Extra args (the fold selection) are forwarded verbatim, so a fold applies to
# whichever stage(s) you point it at.
run_stage1() {           # run_stage1 <out_dir> [extra args...]
    local OUT="$1"; shift
    python scripts/train_ldm_7tcdm3d.py \
        --stage 1 --out_dir "${OUT}" --data_root fmri --epochs 200 \
        --seed ${SEED} "$@"
}
run_stage2() {           # run_stage2 <out_dir> <ae_ckpt> [extra args...]
    local OUT="$1"; local AE="$2"; shift 2
    python scripts/train_ldm_7tcdm3d.py \
        --stage 2 --ae_ckpt "${AE}" --out_dir "${OUT}" --data_root fmri --epochs 2000 \
        --seed ${SEED} "$@"
}
run_two_stage() {        # run_two_stage <out_dir> [extra args...]  (AE from same dir)
    local OUT="$1"; shift
    echo "========================================================================"
    echo " ${OUT}"
    echo "========================================================================"
    run_stage1 "${OUT}" "$@"
    run_stage2 "${OUT}" "${OUT}/stage1_best.pt" "$@"
    echo "Done. Best checkpoint: ${OUT}/stage2_best.pt"
}

# ── modes ─────────────────────────────────────────────────────────────────────
if [ "${MODE}" = "holdout" ]; then
    # Legacy: single random val_frac=0.15 holdout (200/35), no k-fold anywhere.
    echo "Mode: legacy holdout (no k-fold)"
    run_two_stage "runs/${RUN}"
    echo ""
    echo "Holdout run complete. Eval with:"
    echo "  python scripts/eval_ldm_7tcdm3d.py --ckpt runs/${RUN}/stage2_best.pt --val_only"

elif [ "${MODE}" = "cv2" ]; then
    # Train the AE once on a holdout split, then CV only stage 2. All stage-2
    # folds reference the one shared AE; the _ae dir is not a _fold* dir, so the
    # aggregator ignores it.
    AE_DIR="runs/${RUN}_ae"
    echo "Mode: AE trained once (${AE_DIR}) + ${N_FOLDS}-fold CV on stage 2 only"
    echo "========================================================================"
    echo " Stage 1 (shared AE)  ->  ${AE_DIR}"
    echo "========================================================================"
    run_stage1 "${AE_DIR}"                       # no --fold => holdout AE
    for FOLD in $(seq 0 $((N_FOLDS - 1))); do
        OUT="runs/${RUN}_fold${FOLD}"
        echo "------------------------------------------------------------------------"
        echo " Stage 2, fold ${FOLD}/${N_FOLDS}  ->  ${OUT}"
        echo "------------------------------------------------------------------------"
        run_stage2 "${OUT}" "${AE_DIR}/stage1_best.pt" --n_folds ${N_FOLDS} --fold ${FOLD}
        echo "Fold ${FOLD} done. Best checkpoint: ${OUT}/stage2_best.pt"
    done
    echo ""
    echo "All stage-2 folds complete. Aggregate the CV metrics with:"
    echo "  python scripts/aggregate_kfold.py runs/${RUN}"

elif [ -n "${MODE}" ]; then
    # Single fold of the k-way split, both stages.
    echo "Mode: single fold ${MODE} of ${N_FOLDS} (both stages)"
    run_two_stage "runs/${RUN}_fold${MODE}" --n_folds ${N_FOLDS} --fold ${MODE}
    echo ""
    echo "Fold ${MODE} complete. Aggregate available folds with:"
    echo "  python scripts/aggregate_kfold.py runs/${RUN}"

else
    # Full k-fold CV: every fold trains both stages on its own split.
    echo "Mode: ${N_FOLDS}-fold cross-validation (both stages)"
    for FOLD in $(seq 0 $((N_FOLDS - 1))); do
        run_two_stage "runs/${RUN}_fold${FOLD}" --n_folds ${N_FOLDS} --fold ${FOLD}
    done
    echo ""
    echo "All folds complete. Aggregate the CV metrics with:"
    echo "  python scripts/aggregate_kfold.py runs/${RUN}"
fi
