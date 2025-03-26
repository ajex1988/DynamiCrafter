import os
import argparse
import shutil


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_dir', type=str, required=True)
    parser.add_argument('--img_ext', type=str, default='png')
    parser.add_argument('--out_dir', type=str, required=True)
    args = parser.parse_args()
    return args


def main():
    """
    Process the HR dataset
    1. To ignore first two frames
    2. To skip db file
    """
    args = parse_args()
    dir_name_list = os.listdir(args.in_dir)

    out_dir = args.out_dir
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    for dir_name in dir_name_list:
        dir_path = os.path.join(args.in_dir, dir_name)
        file_name_list = os.listdir(dir_path)
        img_name_list = []
        for file_name in file_name_list:
            if file_name.endswith(args.img_ext):
                img_name_list.append(file_name)
        img_name_list.sort()
        del img_name_list[:2]

        out_subdir = os.path.join(out_dir, dir_name)
        if not os.path.exists(out_subdir):
            os.makedirs(out_subdir)

        for img_name in img_name_list:
            img_src = os.path.join(dir_path, img_name)
            img_tgt = os.path.join(out_subdir, img_name)
            shutil.copyfile(img_src, img_tgt)


if __name__ == '__main__':
    main(

    )