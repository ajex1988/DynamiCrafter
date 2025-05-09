import os
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_dir', type=str, default='celeba')
    parser.add_argument('--out_dir', type=str, default='celeba_split')
    parser.add_argument('--subset_size', type=int, default=10)
    parser.add_argument('--prompt_file_path', type=str, default="")
    args = parser.parse_args()
    return args


def split_celeba(in_dir, out_dir, subset_size, prompt_file_path):
    img_name_list = os.listdir(in_dir)
    img_name_list.sort()
    img_num = len(img_name_list)
    subset_num = img_num // subset_size
    for i in range(subset_num):
        subset_img_list = img_name_list[i * subset_size: (i + 1) * subset_size]
        subset_dir = os.path.join(out_dir, str(i))
        if not os.path.exists(subset_dir):
            os.makedirs(subset_dir)
        # "copy" the image but creating symbolic link
        for img_name in subset_img_list:
            src_path = os.path.join(in_dir, img_name)
            dst_path = os.path.join(subset_dir, img_name)
            os.symlink(src_path, dst_path)
        # "copy" the prompt file
        _, prompt_file_name = os.path.split(prompt_file_path)
        os.symlink(prompt_file_path, os.path.join(subset_dir, prompt_file_name))


def main():
    args = parse_args()
    split_celeba(args.in_dir, args.out_dir, args.subset_size, args.prompt_file_path)


if __name__ == '__main__':
    main()
