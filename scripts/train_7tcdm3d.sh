#!/bin/bash
# Train the 7TCDM-3D latent diffusion model (KL-autoencoder then latent denoiser).
#
# Usage:
#   bash scripts/train_7tcdm3d.sh [<name>] [<mode>]
#
#   <name>  base run label; outputs go under runs/<name>*. Omit to auto-generate
#           a timestamped name (runs/ldm_7tcdm3d_<ts>*).
#   <mode>  one of:
#     (omitted)  full k-fold CV: every fold trains BOTH stages on its 6/7 split
#     cv2        train the AE ONCE (holdout), then k-fold CV on stage 2 only  [recommended]
#     <int>      only that fold, both stages (resume a crash / one GPU per fold)
#     holdout    the original single 15% holdout (200/35), NO k-fold
#
#   A bare mode keyword works too: `bash scripts/train_7tcdm3d.sh cv2` auto-names
#   the run (so it does NOT become a run literally named "cv2").
#
# Reuse an existing AE for cv2 (skip stage 1 entirely) via the AE_CKPT env var:
#   AE_CKPT=runs/foo/stage1_best.pt bash scripts/train_7tcdm3d.sh myexp cv2
#
# Examples:
#   bash scripts/train_7tcdm3d.sh my_experiment cv2      # AE once + CV stage 2
#   bash scripts/train_7tcdm3d.sh my_experiment          # full CV, both stages
#   bash scripts/train_7tcdm3d.sh my_experiment 3        # only fold 3, both stages
#   bash scripts/train_7tcdm3d.sh my_experiment holdout  # legacy single holdout
set -e

RUN="${1:-}"
MODE="${2:-}"
N_FOLDS=7
SEED=42

# If the 1st arg is actually a mode keyword and no 2nd arg was given, the user
# omitted the run name — treat it as the mode and auto-name the run. Prevents
# `bash train_7tcdm3d.sh cv2` from silently becoming a run *named* "cv2" in the
# default (full-CV) mode.
if [ -z "${MODE}" ]; then
    case "${RUN}" in
        cv2|holdout) MODE="${RUN}"; RUN="" ;;
    esac
fi
RUN="${RUN:-ldm_7tcdm3d_$(date +%Y-%m-%d_%H-%M-%S)}"

# ── stage runners ─────────────────────────────────────────────────────────────
# Extra args (the fold selection) are forwarded verbatim to whichever stage.
run_stage1() {           # run_stage1 <out_dir> [extra args...]
    local OUT="$1"; shift
    python scripts/train_ldm_7tcdm3d.py \
        --stage 1 --out_dir "${OUT}" --data_root fmri --epochs 200 \
        --wandb_group "${RUN}" --seed ${SEED} "$@"
}
run_stage2() {           # run_stage2 <out_dir> <ae_ckpt> [extra args...]
    local OUT="$1"; local AE="$2"; shift 2
    python scripts/train_ldm_7tcdm3d.py \
        --stage 2 --ae_ckpt "${AE}" --out_dir "${OUT}" --data_root fmri --epochs 2000 \
        --wandb_group "${RUN}" --seed ${SEED} "$@"
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

echo "Run name: ${RUN}   |   Mode: ${MODE:-full-cv}"

# ── modes ─────────────────────────────────────────────────────────────────────
if [ "${MODE}" = "holdout" ]; then
    # Legacy: single random val_frac=0.15 holdout (200/35), no k-fold anywhere.
    echo "Mode: legacy holdout (no k-fold)"
    run_two_stage "runs/${RUN}"
    echo ""
    echo "Holdout run complete. Eval with:"
    echo "  python scripts/eval_ldm_7tcdm3d.py --ckpt runs/${RUN}/stage2_best.pt --val_only"

elif [ "${MODE}" = "cv2" ]; then
    # Train the AE once (or reuse AE_CKPT), then CV only stage 2. All stage-2
    # folds reference the one shared AE; the _ae dir is not a _fold* dir, so the
    # aggregator ignores it.
    if [ -n "${AE_CKPT}" ]; then
        AE="${AE_CKPT}"
        if [ ! -f "${AE}" ]; then
            echo "ERROR: AE_CKPT not found: ${AE}" >&2; exit 1
        fi
        echo "Mode: ${N_FOLDS}-fold CV on stage 2, reusing AE: ${AE} (stage 1 skipped)"
    else
        AE_DIR="runs/${RUN}_ae"
        echo "Mode: AE trained once (${AE_DIR}) + ${N_FOLDS}-fold CV on stage 2"
        echo "========================================================================"
        echo " Stage 1 (shared AE)  ->  ${AE_DIR}"
        echo "========================================================================"
        run_stage1 "${AE_DIR}"                       # no --fold => holdout AE
        AE="${AE_DIR}/stage1_best.pt"
    fi
    for FOLD in $(seq 0 $((N_FOLDS - 1))); do
        OUT="runs/${RUN}_fold${FOLD}"
        echo "------------------------------------------------------------------------"
        echo " Stage 2, fold ${FOLD}/${N_FOLDS}  ->  ${OUT}"
        echo "------------------------------------------------------------------------"
        run_stage2 "${OUT}" "${AE}" --n_folds ${N_FOLDS} --fold ${FOLD}
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
