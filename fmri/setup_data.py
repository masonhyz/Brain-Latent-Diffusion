"""
setup_data.py

Populate this fmri/ folder from the raw reconstructed source on disk. For each
subject that has BOTH a pre- and a post-surgery single-delay CBF map, copy them
into pre_surgery/ and 6_months_post_surgery/ using the <year>_<id>.nii.gz naming
(see README.md). Subjects missing either side are skipped and reported.

This script lives in fmri/. All paths are anchored to this file's location, so
it works no matter which directory you run it from:

  python fmri/setup_data.py
"""

import shutil
from pathlib import Path

# fmri/ is the destination folder this script sits in; the raw source lives at
# /data/mosszhao (two levels above the repo root).
FMRI_DIR = Path(__file__).resolve().parent            # <repo>/fmri
REPO_ROOT = FMRI_DIR.parent                            # <repo>
DATA_ROOT = REPO_ROOT.parent.parent / "data" / "mosszhao"  # e.g. /data/mosszhao


if __name__ == "__main__":
    folder_pre = FMRI_DIR / "pre_surgery"
    folder_post = FMRI_DIR / "6_months_post_surgery"

    folder_pre.mkdir(parents=True, exist_ok=True)
    folder_post.mkdir(parents=True, exist_ok=True)

    for year in [2024]:
        for subject_id in range(1, 100):
            subject_id = str(subject_id).zfill(3)
            subj = DATA_ROOT / f"moyamoya_{year}_nifti" / f"moyamoya_stanford_{year}_{subject_id}"
            path_pre = subj / "derived/pre_surgery_yes_diamox/perf/asl_single_delay_pre_diamox/CBF_Single_Delay_Pre_Diamox_standard_nonlin.nii.gz"
            path_post = subj / "derived/post_surgery_yes_diamox_1/perf/asl_single_delay_pre_diamox/CBF_Single_Delay_Pre_Diamox_standard_nonlin.nii.gz"

            pre_exists = path_pre.exists()
            post_exists = path_post.exists()
            if pre_exists and post_exists:
                shutil.copy2(path_pre, folder_pre / f"{year}_{subject_id}.nii.gz")
                shutil.copy2(path_post, folder_post / f"{year}_{subject_id}.nii.gz")
            else:
                if not pre_exists:
                    print(f"{year}_{subject_id} pre not found")
                if not post_exists:
                    print(f"{year}_{subject_id} post not found")
