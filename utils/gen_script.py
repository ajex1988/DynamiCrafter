import os
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--template", type=str, help="template script")
    parser.add_argument("--in_dir", type=str, help="input directory")
    parser.add_argument("--out_dir", type=str, help="output directory")
    parser.add_argument("--out_file", type=str, help="output file")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    dir_name_list = os.listdir(args.in_dir)
    dir_name_list.sort()

    with open(args.out_file, "w") as f:
        for dir_name in dir_name_list:
            input_dir = os.path.join(args.in_dir, dir_name)
            output_dir = os.path.join(args.out_dir, dir_name)
            script = args.template.replace("{0}", input_dir)
            script = script.replace("{1}", output_dir)
            f.write(script+"\n")


if __name__ == "__main__":
    main()
