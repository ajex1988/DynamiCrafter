import argparse
import os
import random
import shutil


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_dir', type=str, help='input directory')
    parser.add_argument('--out_dir', type=str, help='output directory')
    parser.add_argument('--ratio', type=float, default=9.0, help='ratio of train and val')
    return parser.parse_args()


def main():
    args = parse_args()
    img_dir = os.path.join(args.in_dir, 'img_dir')
    ann_dir = os.path.join(args.in_dir, 'ann_dir')
    img_name_list = os.listdir(img_dir)
    random.seed(10)
    random.shuffle(img_name_list)
    train_ratio = args.ratio/(1.0+args.ratio)
    train_num = int(len(img_name_list) * train_ratio)
    train_name_list = img_name_list[:train_num]
    val_name_list = img_name_list[train_num:]

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    out_img_dir = os.path.join(args.out_dir, 'img_dir')
    train_img_dir = os.path.join(out_img_dir, 'train')
    val_img_dir = os.path.join(out_img_dir, 'val')

    if not os.path.exists(train_img_dir):
        os.makedirs(train_img_dir)
    if not os.path.exists(val_img_dir):
        os.makedirs(val_img_dir)

    out_ann_dir = os.path.join(args.out_dir, 'ann_dir')
    train_ann_dir = os.path.join(out_ann_dir, 'train')
    val_ann_dir = os.path.join(out_ann_dir, 'val')

    if not os.path.exists(train_ann_dir):
        os.makedirs(train_ann_dir)
    if not os.path.exists(val_ann_dir):
        os.makedirs(val_ann_dir)

    for train_img_name in train_name_list:
        train_img_id = os.path.splitext(train_img_name)[0]
        train_img_path_src = os.path.join(img_dir, train_img_name)
        train_img_path_dst = os.path.join(train_img_dir, train_img_name)
        shutil.copyfile(train_img_path_src, train_img_path_dst)

        train_ann_path_src = os.path.join(ann_dir, train_img_id+".png")
        train_ann_path_dst = os.path.join(train_ann_dir, train_img_id+".png")
        shutil.copyfile(train_ann_path_src, train_ann_path_dst)

    for val_img_name in val_name_list:
        val_img_id = os.path.splitext(val_img_name)[0]
        val_img_path_src = os.path.join(img_dir, val_img_name)
        val_img_path_dst = os.path.join(val_img_dir, val_img_name)
        shutil.copyfile(val_img_path_src, val_img_path_dst)

        val_ann_path_src = os.path.join(ann_dir, val_img_id+".png")
        val_ann_path_dst = os.path.join(val_ann_dir, val_img_id+".png")
        shutil.copyfile(val_ann_path_src, val_ann_path_dst)


if __name__ == '__main__':
    main()
