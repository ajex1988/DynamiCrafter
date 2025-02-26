import os
import random
import torch

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from decord import VideoReader, cpu
from tqdm import tqdm

class YouHQ(Dataset):
    """
    Dataset for YouHQ
    https://arxiv.org/pdf/2312.06640
    Structure of YouHQ Dataset:
        YouHQ-Train
            animal
                subdir_1
                    video1.mp4
                    video2.mp4
                    ...
                subdir_2
                ...
            close
            distant
            ...
        YouHQ-Val
        ...
    The original dataset does not have caption.
    We can use I2T models to automatically generate captions.
    We assume the captions follow the same structure ($video_name$.txt).
    """
    def __init__(self,
                 video_dir,
                 caption_dir=None,
                 video_length=16,
                 resolution=(512, 512),
                 frame_stride=1,
                 frame_stride_min=1,
                 spatial_transform_type="resize_center_crop",
                 crop_resolution=None,
                 fps_max=None,
                 fixed_fps=None,
                 random_fs=False):
        self.caption_dir = caption_dir
        self.video_length = video_length
        self.resolution = resolution
        self.frame_stride = frame_stride
        self.frame_stride_min = frame_stride_min
        self.crop_resolution = crop_resolution
        self.fps_max = fps_max
        self.fixed_fps = fixed_fps
        self.random_fs = random_fs
        self.spatial_transform_type = spatial_transform_type

        ## Data preprocessing
        if spatial_transform_type is not None:
            if spatial_transform_type == "random_crop":
                self.spatial_transform = transforms.RandomCrop(crop_resolution)
            elif spatial_transform_type == "center_crop":
                self.spatial_transform = transforms.Compose([
                    transforms.CenterCrop(resolution),
                    ])
            elif spatial_transform_type == "resize_center_crop":
                self.spatial_transform = transforms.Compose([
                    transforms.Resize(min(self.resolution)),
                    transforms.CenterCrop(self.resolution),
                    ])
            elif spatial_transform_type == "resize":
                self.spatial_transform = transforms.Resize(self.resolution)
            else:
                raise NotImplementedError

        ## Dataset loading
        print("Loading YouHQ dataset...")
        self.video_path_list = []
        category_name_list = os.listdir(video_dir)
        for category_name in category_name_list:
            category_subdir = os.path.join(video_dir, category_name)
            if os.path.isdir(category_subdir):
                video_dir_name_list = os.listdir(category_subdir)
                for video_dir_name in video_dir_name_list:
                    video_dir_path = os.path.join(category_subdir, video_dir_name)
                    if os.path.isdir(video_dir_path):
                        video_name_list = os.listdir(video_dir_path)
                        for video_name in video_name_list:
                            video_path = os.path.join(video_dir_path, video_name)
                            if video_path.endswith(".mp4"):
                                self.video_path_list.append(video_path)
        print(f"Found {len(self.video_path_list)} YouHQ videos.")
        if caption_dir is not None:
            print("Loading YouHQ captions...")
            ## TODO
            print("Captions loaded")



    def __len__(self):
        return len(self.video_path_list)

    def __getitem__(self, index):
        ## If use random frame stride, randomly sample a fs in [min_fs, fs]
        if self.random_fs:
            frame_stride = random.randint(self.frame_stride_min, self.frame_stride)
        else:
            frame_stride = self.frame_stride

        video_path = self.video_path_list[index]
        video_reader = VideoReader(video_path, ctx=cpu(0))
        if len(video_reader) < self.video_length:
            raise Exception(f"Not enough frames in video {video_path}")

        frame_num_needed = frame_stride * (self.video_length-1) + 1
        if len(video_reader) < frame_num_needed:
            raise Exception(f"Frame stride({frame_stride}) is too large, consider reducing it")

        random_frame_range = len(video_reader) - frame_num_needed
        start_frame = random.randint(0, random_frame_range)
        frame_indices = [start_frame + i*frame_stride for i in range(self.video_length)]

        frames = video_reader.get_batch(frame_indices)

        # numpy array to pytorch tensor
        frames = torch.tensor(frames.asnumpy()).float()
        # [f h w c] -> [c f h w]
        frames = frames.permute(3, 0, 1, 2)

        # transforms
        if self.spatial_transform_type is not None:
            frames = self.spatial_transform(frames)

        # normalize to [-1, 1]
        frames = (frames / 255 - 0.5) * 2

        # fps
        fps_ori = video_reader.get_avg_fps()
        fps_clip = fps_ori // frame_stride
        if self.fps_max is not None and fps_clip > self.fps_max:
            fps_clip = self.fps_max

        sample = {'video': frames, 'caption': "", 'path': video_path, 'fps': fps_clip, 'frame_stride': frame_stride}
        return sample


if __name__ == "__main__":
    """
    Test the YouHQ dataset.
    """
    video_dir = "/workspace/shared-dir/zzhu/data/VSR/YouHQ-Train/YouHQ-Train"
    log_dir = "/workspace/shared-dir/zzhu/tmp/20250224"
    dataset = YouHQ(video_dir=video_dir,)
    dataloader = DataLoader(dataset,
                            batch_size=1,
                            num_workers=0,
                            shuffle=False)
    info = {}
    with open(os.path.join(log_dir, "log.txt"), "w") as writer:
        for i, batch in tqdm(enumerate(dataloader), desc="Data Batch"):
            video_path = batch['path'][0]
            b, c, f, h, w = batch['video'].shape[0], batch['video'].shape[1], batch['video'].shape[2], batch['video'].shape[3], batch['video'].shape[4]
            writer.write(f"{video_path} {b} {c} {f} {h} {w}\n")
            k = f"{b}_{c}_{f}_{h}_{w}"
            if k not in info:
                info[k] = 1
            else:
                info[k] += 1
    print(info)