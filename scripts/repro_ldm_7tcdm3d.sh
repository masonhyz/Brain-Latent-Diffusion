#!/bin/bash
set -e

python scripts/train_ldm_7tcdm3d.py \
    --stage 1 \
    --out_dir runs/repro_ldm_7tcdm3d \
    --ae_res_blocks 2 \
    --seed 42

python scripts/train_ldm_7tcdm3d.py \
    --stage 2 \
    --ae_ckpt runs/repro_ldm_7tcdm3d/stage1_best.pt \
    --out_dir runs/repro_ldm_7tcdm3d \
    --ae_res_blocks 2 \
    --epochs 200 \
    --seed 42
