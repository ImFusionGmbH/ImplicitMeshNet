import os
from glob import glob
import subprocess

import numpy as np
import SimpleITK as sitk
import pandas as pd

from fire import Fire
from tqdm import tqdm

from implicitmeshnet_dataset import resample_image


def main(
    input_dir_lv: str = "test_out_lv",
    input_dir_la: str = "test_out_la",
    output_dir: str = "test_out",
    encrypted_gt_dir: str = "/Users/laves/ImFusion/mmwhs/MMWHS_evaluation_testdata_label_encrypt_1mm_forpublic",
):
    segmentations_lv = sorted(glob(os.path.join(input_dir_lv, "*ct_test_*_mesh_labelmap.nii.gz")))
    segmentations_la = sorted(glob(os.path.join(input_dir_la, "*ct_test_*_mesh_labelmap.nii.gz")))
    # segmentations_lv = sorted(glob(os.path.join(input_dir_lv, "*ct_test_*_unet_segmentation.nii.gz")))
    # segmentations_la = sorted(glob(os.path.join(input_dir_la, "*ct_test_*_unet_segmentation.nii.gz")))

    if len(segmentations_lv) == 0 or len(segmentations_la) == 0:
        raise ValueError(f"No segmentations found in {input_dir_lv} or {input_dir_la}")

    if len(segmentations_lv) != len(segmentations_la):
        raise ValueError(f"Number of segmentations do not match: {len(segmentations_lv)} != {len(segmentations_la)}")

    os.makedirs(output_dir, exist_ok=False)

    pbar = tqdm(segmentations_lv, ncols=80)
    for segmentation_lv, segmentation_la in zip(pbar, segmentations_la):
        img_itk_lv = sitk.ReadImage(segmentation_lv)
        img_itk_la = sitk.ReadImage(segmentation_la)

        img_lv = sitk.GetArrayFromImage(img_itk_lv)
        img_la = sitk.GetArrayFromImage(img_itk_la)

        img_out = np.zeros_like(img_lv).astype(np.uint16)
        img_out[img_la == 1] = 420  # these values come from the MMWHS dataset
        img_out[img_lv == 1] = 500

        img_itk_out = sitk.GetImageFromArray(img_out)
        img_itk_out.CopyInformation(img_itk_lv)

        output_fname = os.path.join(output_dir, os.path.basename(segmentation_lv).replace("_mesh_labelmap", "_label"))
        sitk.WriteImage(img_itk_out, output_fname)

        # resample to 1mm spacing
        img_itk_out = resample_image(img_itk_out, (1, 1, 1), sitk.sitkNearestNeighbor)
        sitk.WriteImage(img_itk_out, output_fname.replace(".nii.gz", "_1mm.nii.gz"))

        # call:
        # wine zxhCardWhsEvaluate.exe \
        # /Users/laves/Desktop/doccunet/test_out/ct_test_2018_label_1mm.nii.gz \
        # /Users/laves/ImFusion/mmwhs/MMWHS_evaluation_testdata_label_encrypt_1mm_forpublic/nii/ct_test_2018_label_encrypt_1mm.nii.gz \
        # 0 ALL dice.csv ct 2018 --decodeseg2

        ct_num = os.path.basename(segmentation_lv).split("_")[2]
        subprocess.run(
            [
                "/opt/homebrew/bin/wine",
                os.path.join(encrypted_gt_dir, "zxhCardWhsEvaluate.exe"),
                output_fname.replace(".nii.gz", "_1mm.nii.gz"),
                os.path.join(
                    encrypted_gt_dir,
                    "nii",
                    f"ct_test_{ct_num}_label_encrypt_1mm.nii.gz",
                ),
                "0",
                "ALL",
                os.path.join(output_dir, "dice.csv"),
                "ct",
                ct_num,
                "--decodeseg2",
            ],
            check=True
        )

    dice_df = pd.read_csv(os.path.join(output_dir, "dice.csv"), header=None, sep="\t")
    print("Dice_la:", dice_df[3].mean(), "±", dice_df[3].std())
    print("Dice_lv:", dice_df[0].mean(), "±", dice_df[0].std())


if __name__ == "__main__":
    Fire(main)
