from matlab_resize import matlab_imresize
import argparse
import os
import PIL.Image as Image
import numpy as np
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_dir", type=str)
    parser.add_argument("--out_dir", type=str)
    parser.add_argument("--in_ext", type=str, default="png")
    parser.add_argument("--out_ext", type=str, default="png")
    parser.add_argument("--in_dim", type=int, default=16)
    parser.add_argument("--out_dim", type=int, default=9)
    args = parser.parse_args()
    return args


def main():
    print("Resizing images using matlab implementation")
    args = parse_args()
    in_dir = args.in_dir
    out_dir = args.out_dir

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    img_name_list = os.listdir(in_dir)
    for img_name in tqdm(img_name_list):
        img_path = os.path.join(in_dir, img_name)
        out_path = os.path.join(out_dir, img_name)

        img = Image.open(img_path)
        img = np.array(img)

        img_h, img_w = img.shape[:2]
        img_h_resized = img_h * args.out_dim // args.in_dim
        img_w_resized = img_w * args.out_dim // args.in_dim
        out_size = (img_h_resized, img_w_resized)

        img = img.astype(np.float32)
        img_resized = matlab_imresize(img, outsize=out_size)
        img_resized = img_resized.astype(np.uint8)

        img_resized = Image.fromarray(img_resized)

        img_resized.save(out_path)



if __name__ == '__main__':
    main()
