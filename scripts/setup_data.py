import os
import shutil


if __name__ == "__main__":
    folder_pre = os.path.join("fmri", "pre_surgery")
    folder_post = os.path.join("fmri", "6_months_post_surgery")

    os.makedirs(folder_pre, exist_ok=True)
    os.makedirs(folder_post, exist_ok=True)

    for year in [2024]:
        for subject_id in range(1, 100):
            subject_id = str(subject_id).zfill(3)
            path_pre = f"../../data/mosszhao/moyamoya_{year}_nifti/moyamoya_stanford_{year}_{subject_id}/derived/pre_surgery_yes_diamox/perf/asl_single_delay_pre_diamox/CBF_Single_Delay_Pre_Diamox_standard_nonlin.nii.gz"
            path_post = f"../../data/mosszhao/moyamoya_{year}_nifti/moyamoya_stanford_{year}_{subject_id}/derived/post_surgery_yes_diamox_1/perf/asl_single_delay_pre_diamox/CBF_Single_Delay_Pre_Diamox_standard_nonlin.nii.gz"

            pre_exists = os.path.exists(path_pre)
            post_exists = os.path.exists(path_post)
            if pre_exists and post_exists:
                shutil.copy2(path_pre, os.path.join(folder_pre, f"{year}_{subject_id}.nii.gz"))
                shutil.copy2(path_post, os.path.join(folder_post, f"{year}_{subject_id}.nii.gz"))
            else:
                if not pre_exists:
                    print(f"{year}_{subject_id} pre not found")
                if not post_exists:
                    print(f"{year}_{subject_id} post not found")
            
