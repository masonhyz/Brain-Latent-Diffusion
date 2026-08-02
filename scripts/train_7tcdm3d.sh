#!/bin/bash
# Train the 7TCDM-3D latent diffusion model (KL-autoencoder then latent denoiser).
#
# Usage:
#   bash scripts/train_7tcdm3d.sh [<name>] [<mode>]
#
#   <name>  base run label; outputs go under runs/<name>*. Omit to identify the
#           run purely by timestamp (runs/<ts>*) — no name needed.
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
RUN="${RUN:-$(date +%Y-%m-%d_%H-%M-%S)}"

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
        --stage 2 --ae_ckpt "${AE}" --out_dir "${OUT}" --data_root fmri --epochs 1000 \
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

# Echo a stage-1 AE checkpoint path on stdout, training one if needed. Reuses
# ``AE_CKPT`` when set (stage 1 skipped); otherwise trains the shared AE once
# into ``runs/<RUN>_stage1_ae``. All human-readable output goes to stderr so the
# command substitution ``$(ensure_ae)`` captures only the checkpoint path.
ensure_ae() {
    if [ -n "${AE_CKPT}" ]; then
        [ -f "${AE_CKPT}" ] || { echo "ERROR: AE_CKPT not found: ${AE_CKPT}" >&2; exit 1; }
        echo "Reusing AE: ${AE_CKPT} (stage 1 skipped)" >&2
        echo "${AE_CKPT}"
        return
    fi
    local AE_DIR="runs/${RUN}_stage1_ae"
    echo "========================================================================" >&2
    echo " Stage 1 (shared AE)  ->  ${AE_DIR}" >&2
    echo "========================================================================" >&2
    run_stage1 "${AE_DIR}" 1>&2                   # no --fold => holdout AE
    echo "${AE_DIR}/stage1_best.pt"
}

echo "Run name: ${RUN}   |   Mode: ${MODE:-full-cv}"

# ── modes ─────────────────────────────────────────────────────────────────────
# Layout: one run == one <RUN> id (a timestamp unless you named it). Stage 1
# (rarely retrained) lives in runs/<RUN>_stage1_ae; stage 2 in its own sibling
# runs/<RUN>_stage2*, with k-fold folds nested under runs/<RUN>_stage2_cv/fold*.
if [ "${MODE}" = "holdout" ]; then
    # Single random val_frac=0.15 holdout (200/35), no k-fold anywhere.
    echo "Mode: holdout (no k-fold)"
    AE="$(ensure_ae)"
    S2_DIR="runs/${RUN}_stage2"
    echo "========================================================================"
    echo " Stage 2 (holdout)  ->  ${S2_DIR}"
    echo "========================================================================"
    run_stage2 "${S2_DIR}" "${AE}"
    echo ""
    echo "Holdout run complete. Eval with:"
    echo "  python scripts/eval_ldm_7tcdm3d.py --ckpt ${S2_DIR}/stage2_best.pt --val_only"

elif [ "${MODE}" = "cv2" ]; then
    # Train the AE once (or reuse AE_CKPT), then CV only stage 2. All stage-2
    # folds reference the one shared AE and nest under a single _cv parent.
    echo "Mode: ${N_FOLDS}-fold CV on stage 2 (AE trained once / reused)"
    AE="$(ensure_ae)"
    S2_PARENT="runs/${RUN}_stage2_cv"
    for FOLD in $(seq 0 $((N_FOLDS - 1))); do
        OUT="${S2_PARENT}/fold${FOLD}"
        echo "------------------------------------------------------------------------"
        echo " Stage 2, fold ${FOLD}/${N_FOLDS}  ->  ${OUT}"
        echo "------------------------------------------------------------------------"
        run_stage2 "${OUT}" "${AE}" --n_folds ${N_FOLDS} --fold ${FOLD}
        echo "Fold ${FOLD} done. Best checkpoint: ${OUT}/stage2_best.pt"
    done
    echo ""
    echo "All stage-2 folds complete. Aggregate the CV metrics with:"
    echo "  python scripts/aggregate_kfold.py ${S2_PARENT}"

elif [ -n "${MODE}" ]; then
    # Single fold of the k-way split, both stages, nested under a _cv parent so
    # more folds can be added to the same run later and aggregated together.
    echo "Mode: single fold ${MODE} of ${N_FOLDS} (both stages)"
    PARENT="runs/${RUN}_cv"
    run_two_stage "${PARENT}/fold${MODE}" --n_folds ${N_FOLDS} --fold ${MODE}
    echo ""
    echo "Fold ${MODE} complete. Aggregate available folds with:"
    echo "  python scripts/aggregate_kfold.py ${PARENT}"

else
    # Full k-fold CV: every fold trains both stages on its own split, nested
    # under one parent.
    echo "Mode: ${N_FOLDS}-fold cross-validation (both stages)"
    PARENT="runs/${RUN}_cv"
    for FOLD in $(seq 0 $((N_FOLDS - 1))); do
        run_two_stage "${PARENT}/fold${FOLD}" --n_folds ${N_FOLDS} --fold ${FOLD}
    done
    echo ""
    echo "All folds complete. Aggregate the CV metrics with:"
    echo "  python scripts/aggregate_kfold.py ${PARENT}"
fi
