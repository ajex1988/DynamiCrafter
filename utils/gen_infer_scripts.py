import os
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=str)
    parser.add_argument('--out_dir', type=str)
    parser.add_argument('--script_out_dir', type=str)
    parser.add_argument('--n_split', type=int, default=3)
    args = parser.parse_args()
    return args


def gen_scripts(args):
    subset_name_list = os.listdir(args.dataset_dir)
    subset_name_list.sort()

    script_content_list = [""] * args.n_split
    for i, subset_name in enumerate(subset_name_list):
        in_dir = os.path.join(args.dataset_dir, subset_name)
        out_dir = os.path.join(args.out_dir, subset_name)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        script_content = f"CUDA_VISIBLE_DEVICES={i % args.n_split} python3 scripts/evaluation/inference_save_frames.py --ckpt_path /workspace/shared-dir/zzhu/projects/DynamiCrafter/checkpoints/dynamicrafter_1024_v1/model.ckpt  --config /workspace/shared-dir/zzhu/projects/DynamiCrafter/configs/inference_1024_v1.0.yaml --savedir {out_dir} --n_samples 1 --bs 1 --height 1024 --width 1024 --unconditional_guidance_scale 7.5 --ddim_steps 50 --ddim_eta 1.0 --prompt_dir {in_dir} --text_input --video_length 16 --frame_stride 30 --timestep_spacing uniform_trailing --guidance_rescale 0.7"
        script_content_list[i % args.n_split] += script_content + "\n"

    for i in range(args.n_split):
        out_file_path = os.path.join(args.script_out_dir, f"inference_save_frames_{i}.sh")
        with open(out_file_path, "w") as f:
            f.write(script_content_list[i])


def main():
    args = parse_args()
    gen_scripts(args)

if __name__ == "__main__":
    main()