import os
import shutil


if __name__ == "__main__":
    # clear the folder fmri
    if os.path.exists("fmri"):
        shutil.rmtree("fmri")

    # make the data folder if not exist
    if not os.path.exists("fmri"):
        os.makedirs("fmri")

    folder_pre = os.path.join("fmri", "pre_surgery")
    folder_post = os.path.join("fmri", "6_months_post_surgery")

    # make folders if not exist
    if not os.path.exists(folder_pre):
        os.makedirs(folder_pre)
    if not os.path.exists(folder_post):
        os.makedirs(folder_post)
    
    for year in [2020, 2021, 2022, 2023]:
        for subject_id in range(1, 100):
            subject_id = str(subject_id).zfill(3)
            path_pre = f"../../data/mosszhao/moyamoya_{year}_nifti/moyamoya_stanford_{year}_{subject_id}/derived/pre_surgery_yes_diamox/perf/asl_single_delay_pre_diamox/CBF_Single_Delay_Pre_Diamox_standard_nonlin.nii.gz"
            path_post = f"../../data/mosszhao/moyamoya_{year}_nifti/moyamoya_stanford_{year}_{subject_id}/derived/post_surgery_yes_diamox_1/perf/asl_single_delay_pre_diamox/CBF_Single_Delay_Pre_Diamox_standard_nonlin.nii.gz"
            
            if os.path.exists(path_pre) and os.path.exists(path_post):
                shutil.copy2(path_pre, os.path.join(folder_pre, f"{year}_{subject_id}.nii.gz"))
                shutil.copy2(path_post, os.path.join(folder_post, f"{year}_{subject_id}.nii.gz"))
            else:
                print(f"{year}_{subject_id} not found")
            
