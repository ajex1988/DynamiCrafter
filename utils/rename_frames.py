import os
import argparse
import shutil


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', required=True, help='input directory')
    parser.add_argument('--img_format', type=str, default='png', help='image format')
    parser.add_argument('--output_dir', required=True, help='output directory')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    input_dir = args.input_dir
    output_dir = args.output_dir
    img_format = args.img_format

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    img_name_list = os.listdir(input_dir)
    img_name_list = sorted(img_name_list)

    img_name_list_filtered = []

    for i, img_name in enumerate(img_name_list):
        if img_name.endswith(img_format):
            img_name_list_filtered.append(img_name)

    for i, img_name in enumerate(img_name_list_filtered):
        img_path_src = os.path.join(input_dir, img_name)
        img_path_dst = os.path.join(output_dir, f"{i:05}.{img_format}")
        shutil.copyfile(img_path_src, img_path_dst)



if __name__ == "__main__":
    main()
