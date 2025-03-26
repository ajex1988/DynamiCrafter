import shutil
import os
import pandas as pd
import argparse
import numpy as np
from PIL import Image, ImageOps


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_dir", type=str, help="Input image directory")
    parser.add_argument("--in_img_ext", type=str, default="jpg", help="Input image format")
    parser.add_argument("--out_dir", type=str, help="Output directory")
    parser.add_argument("--out_ds_name", type=str, default="keye_fab_liq", help="Output dataset name")
    parser.add_argument("--vis_dir", type=str, default=None, help="Visualization directory for mask checking")

    args = parser.parse_args()
    return args


def get_color_img(img, color=(255, 0, 0)):
    pure_color_img = Image.new("RGB", img.size, color)
    return pure_color_img


def load_csv_as_img(csv_file_path):
    df = pd.read_csv(csv_file_path)
    img = df.astype(int).to_numpy()
    return img


def main():
    """
    Generate standard segmentation dataset for fab liq segmentation.
    Segmentation obtained from leaf's annotation tool based on sam. Segmentation file has the same name as image but csv format.
    This script convert the image & segmentation mask to standard segmentation dataset format so that SOTA models can be trained for evaluation.
    Input format:
        img_dir
            img_1
            img_2
            ...
        mask_dir
            img_1
                mask_1
                mask_2
                ...
    Output format
        dataset_name
            img_dir
                img_1
                img_2
                ...
            ann_dir
                mask_1
                mask_2
                ...
    Since our task is a binary segmentation problem (segment liquid region from non-liquid background), the output mask should be either 0 () or 1 ().
    Note that train/val splitting has not been done on the output format. That should be handled by a seperate script.
    """
    print("Generating fab liq segmentation dataset...")

    args = parse_args()
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    out_ds_dir = os.path.join(args.out_dir, args.out_ds_name)
    if not os.path.exists(out_ds_dir):
        os.makedirs(out_ds_dir)

    out_img_dir = os.path.join(out_ds_dir, "img_dir")
    if not os.path.exists(out_img_dir):
        os.makedirs(out_img_dir)

    out_ann_dir = os.path.join(out_ds_dir, "ann_dir")
    if not os.path.exists(out_ann_dir):
        os.makedirs(out_ann_dir)

    if args.vis_dir is not None:
        if not os.path.exists(args.vis_dir):
            os.makedirs(args.vis_dir)
        vis_ds_dir = os.path.join(args.vis_dir, args.out_ds_name)
        if not os.path.exists(vis_ds_dir):
            os.makedirs(vis_ds_dir)

    img_name_list = os.listdir(args.in_dir)
    for img_name in img_name_list:
        if img_name.endswith(args.in_img_ext):
            img_id, _ = os.path.splitext(img_name)
            mask_file_name = img_name + "_anno_sam.csv"
            mask_file_path = os.path.join(args.in_dir, mask_file_name)
            if os.path.exists(mask_file_path):
                mask = load_csv_as_img(mask_file_path)

                ## Copy image
                src_img_path = os.path.join(args.in_dir, img_name)
                tgt_img_path = os.path.join(out_img_dir, img_name)
                shutil.copyfile(src=src_img_path, dst=tgt_img_path)

                img = Image.open(src_img_path)
                width, height = img.size
                mask = mask.astype(np.uint8)

                ## Save mask
                mask = Image.fromarray(mask)
                mask = mask.resize((width, height))
                out_mask_path = os.path.join(out_ann_dir, img_id + ".png")
                mask.save(out_mask_path)

                ## Save visualization
                if args.vis_dir is not None:
                    img = Image.open(src_img_path)
                    color_img = get_color_img(img)
                    mask = np.array(mask).astype(float) * 0.5
                    mask = mask[:, :, np.newaxis]
                    mask = np.concatenate((mask, mask, mask), axis=2)
                    vis_img = np.array(img) * (1 - np.array(mask).astype(float) * 0.5) + np.array(color_img) * np.array(
                        mask).astype(float) * 0.5
                    vis_img = vis_img.astype(np.uint8)
                    vis_img = Image.fromarray(vis_img)
                    vis_img_path = os.path.join(vis_ds_dir, img_id + ".png")
                    vis_img.save(vis_img_path)





if __name__ == "__main__":
    main()
