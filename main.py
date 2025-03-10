import os
import glob
import argparse
import torch
import time
import torch.nn as nn

import torchvision
import torchvision.transforms as transforms

from tqdm import tqdm
from omegaconf import OmegaConf
from collections import OrderedDict
from utils.utils import instantiate_from_config
from PIL import Image
from einops import rearrange, repeat

from lvdm.models.samplers.ddim import DDIMSampler
from lvdm.models.samplers.ddim_multiplecond import DDIMSampler as DDIMSampler_multicond

from pytorch_lightning import seed_everything


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='interp')
    parser.add_argument('--cfg_path', type=str, default='./configs/inference_512_v1.0.yaml')
    parser.add_argument('--ckpt_path', type=str, default='./checkpoints/dynamicrafter/model.ckpt')
    parser.add_argument('--frame_path_list', nargs="*", type=str, default=[])
    # parser.add_argument('--first_frm_path', type=str, default='')
    # parser.add_argument('--last_frm_path', type=str, default='')
    parser.add_argument('--prompt_file_path', type=str, default='')
    parser.add_argument('--out_dir', type=str, default='')
    parser.add_argument('--height', type=int, default=512)
    parser.add_argument('--width', type=int, default=512)
    parser.add_argument('--frame_stride', type=int, default=3)  # see inference.py
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--video_length', type=int, default=16)
    parser.add_argument("--perframe_ae", action='store_true', default=False,
                        help="if we use per-frame AE decoding, set it to True to save GPU memory, especially for the model of 576x1024")

    # The following argus use the default ones
    parser.add_argument("--n_samples", type=int, default=1, help="num of samples per prompt")
    parser.add_argument('--bs', type=int, default=1)
    parser.add_argument("--ddim_steps", type=int, default=50, help="steps of ddim if positive, otherwise use DDPM")
    parser.add_argument("--ddim_eta", type=float, default=1.0,
                        help="eta for ddim sampling (0.0 yields deterministic sampling)")
    parser.add_argument("--unconditional_guidance_scale", type=float, default=1.0,
                        help="prompt classifier-free guidance")
    parser.add_argument("--negative_prompt", action='store_true', default=False, help="negative prompt")
    parser.add_argument("--text_input", action='store_true', default=False, help="input text to I2V model or not")
    parser.add_argument("--multiple_cond_cfg", action='store_true', default=False,
                        help="use multi-condition cfg or not")
    parser.add_argument("--cfg_img", type=float, default=None, help="guidance scale for image conditioning")
    parser.add_argument("--timestep_spacing", type=str, default="uniform",
                        help="The way the timesteps should be scaled. Refer to Table 2 of the [Common Diffusion Noise Schedules and Sample Steps are Flawed](https://huggingface.co/papers/2305.08891) for more information.")
    parser.add_argument("--guidance_rescale", type=float, default=0.0,
                        help="guidance rescale in [Common Diffusion Noise Schedules and Sample Steps are Flawed](https://huggingface.co/papers/2305.08891)")
    parser.add_argument("--loop", action='store_true', default=False, help="generate looping videos or not")
    parser.add_argument("--interp", action='store_true', default=True,
                        help="generate generative frame interpolation or not")

    parser.add_argument("--interp_type", type=str, default="first_last")
    parser.add_argument("--in_frame_num", type=int, default=5)
    parser.add_argument("--run_on_whole_clip", action='store_true', default=False)
    parser.add_argument("--in_dir", type=str, default="")

    args = parser.parse_args()
    return args


def get_network_description(network):
    if isinstance(network, nn.DataParallel):
        network = network.module
    s = str(network) # network description
    n = sum(map(lambda x: x.numel(), network.parameters())) # network size
    return s, n


def load_model_checkpoint(model, ckpt):
    state_dict = torch.load(ckpt, map_location="cpu")
    if "state_dict" in list(state_dict.keys()):
        state_dict = state_dict["state_dict"]
        try:
            model.load_state_dict(state_dict, strict=True)
        except:
            ## rename the keys for 256x256 model
            new_pl_sd = OrderedDict()
            for k,v in state_dict.items():
                new_pl_sd[k] = v

            for k in list(new_pl_sd.keys()):
                if "framestride_embed" in k:
                    new_key = k.replace("framestride_embed", "fps_embedding")
                    new_pl_sd[new_key] = new_pl_sd[k]
                    del new_pl_sd[k]
            model.load_state_dict(new_pl_sd, strict=True)
    else:
        # deepspeed
        new_pl_sd = OrderedDict()
        for key in state_dict['module'].keys():
            new_pl_sd[key[16:]]=state_dict['module'][key]
        model.load_state_dict(new_pl_sd)
    print('>>> model checkpoint loaded.')
    return model


def get_filelist(data_dir, postfixes):
    patterns = [os.path.join(data_dir, f"*.{postfix}") for postfix in postfixes]
    file_list = []
    for pattern in patterns:
        file_list.extend(glob.glob(pattern))
    file_list.sort()
    return file_list


def get_latent_z(model, videos):
    b, c, t, h, w = videos.shape
    x = rearrange(videos, 'b c t h w -> (b t) c h w')
    z = model.encode_first_stage(x)
    z = rearrange(z, '(b t) c h w -> b c t h w', b=b, t=t)
    return z


def load_prompts(prompt_file):
    f = open(prompt_file, 'r')
    prompt_list = []
    for idx, line in enumerate(f.readlines()):
        l = line.strip()
        if len(l) != 0:
            prompt_list.append(l)
        f.close()
    return prompt_list


def load_frames(frame_path_list, height, width, n_frames, interp_type="first_last"):
    transform = transforms.Compose([
        transforms.Resize((height, width)),
        transforms.CenterCrop((height, width)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])
    if interp_type == "first_last":
        first_frm_path = frame_path_list[0]
        last_frm_path = frame_path_list[-1]
        first_frm = Image.open(first_frm_path).convert('RGB')
        image_tensor1 = transform(first_frm).unsqueeze(1)
        last_frm = Image.open(last_frm_path).convert('RGB')
        image_tensor2 = transform(last_frm).unsqueeze(1)
        frame_tensor1 = repeat(image_tensor1, 'c t h w -> c (repeat t) h w', repeat=n_frames // 2)
        frame_tensor2 = repeat(image_tensor2, 'c t h w -> c (repeat t) h w', repeat=n_frames // 2)
        frame_tensor = torch.cat([frame_tensor1, frame_tensor2], dim=1)
    elif interp_type == "5in_17out":
        ## 5 frames as input and 17 frames as output. 4x in temporal dimension
        frm_1 = Image.open(frame_path_list[0]).convert('RGB')
        frm_2 = Image.open(frame_path_list[1]).convert('RGB')
        frm_3 = Image.open(frame_path_list[2]).convert('RGB')
        frm_4 = Image.open(frame_path_list[3]).convert('RGB')
        frm_5 = Image.open(frame_path_list[4]).convert('RGB')
        img_tensor_1 = transform(frm_1).unsqueeze(1)
        img_tensor_2 = transform(frm_2).unsqueeze(1)
        img_tensor_3 = transform(frm_3).unsqueeze(1)
        img_tensor_4 = transform(frm_4).unsqueeze(1)
        img_tensor_5 = transform(frm_5).unsqueeze(1)
        frm_tensor_1 = repeat(img_tensor_1, 'c t h w -> c (repeat t) h w', repeat=4)
        frm_tensor_2 = repeat(img_tensor_2, 'c t h w -> c (repeat t) h w', repeat=4)
        frm_tensor_3 = repeat(img_tensor_3, 'c t h w -> c (repeat t) h w', repeat=4)
        frm_tensor_4 = repeat(img_tensor_4, 'c t h w -> c (repeat t) h w', repeat=4)
        frame_tensor = torch.cat([frm_tensor_1, frm_tensor_2, frm_tensor_3, frm_tensor_4, img_tensor_5], dim=1)
    elif interp_type == "5in_16out":
        frm_1 = Image.open(frame_path_list[0]).convert('RGB')
        frm_2 = Image.open(frame_path_list[1]).convert('RGB')
        frm_3 = Image.open(frame_path_list[2]).convert('RGB')
        frm_4 = Image.open(frame_path_list[3]).convert('RGB')
        frm_5 = Image.open(frame_path_list[4]).convert('RGB')
        img_tensor_1 = transform(frm_1).unsqueeze(1)
        img_tensor_2 = transform(frm_2).unsqueeze(1)
        img_tensor_3 = transform(frm_3).unsqueeze(1)
        img_tensor_4 = transform(frm_4).unsqueeze(1)
        img_tensor_5 = transform(frm_5).unsqueeze(1)
        frm_tensor_1 = repeat(img_tensor_1, 'c t h w -> c (repeat t) h w', repeat=4)
        frm_tensor_2 = repeat(img_tensor_2, 'c t h w -> c (repeat t) h w', repeat=4)
        frm_tensor_3 = repeat(img_tensor_3, 'c t h w -> c (repeat t) h w', repeat=4)
        frm_tensor_4 = repeat(img_tensor_4, 'c t h w -> c (repeat t) h w', repeat=3)
        frame_tensor = torch.cat([frm_tensor_1, frm_tensor_2, frm_tensor_3, frm_tensor_4, img_tensor_5], dim=1)
    else:
        raise ValueError(f"Unknown interp type: {interp_type}")
    return frame_tensor


def load_prompt(prompt_file):
    with open(prompt_file, 'r') as reader:
        prompt = reader.read()
    return prompt


def load_data_prompts(data_dir, video_size=(256, 256), video_frames=16, interp=False):
    transform = transforms.Compose([
        transforms.Resize(min(video_size)),
        transforms.CenterCrop(video_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])
    ## load prompts
    prompt_file = get_filelist(data_dir, ['txt'])
    assert len(prompt_file) > 0, "Error: found NO prompt file!"
    ###### default prompt
    default_idx = 0
    default_idx = min(default_idx, len(prompt_file) - 1)
    if len(prompt_file) > 1:
        print(f"Warning: multiple prompt files exist. The one {os.path.split(prompt_file[default_idx])[1]} is used.")
    ## only use the first one (sorted by name) if multiple exist

    ## load video
    file_list = get_filelist(data_dir, ['jpg', 'png', 'jpeg', 'JPEG', 'PNG'])
    # assert len(file_list) == n_samples, "Error: data and prompts are NOT paired!"
    data_list = []
    filename_list = []
    prompt_list = load_prompts(prompt_file[default_idx])
    n_samples = len(prompt_list)
    for idx in range(n_samples):
        if interp:
            image1 = Image.open(file_list[2 * idx]).convert('RGB')
            image_tensor1 = transform(image1).unsqueeze(1)  # [c,1,h,w]
            image2 = Image.open(file_list[2 * idx + 1]).convert('RGB')
            image_tensor2 = transform(image2).unsqueeze(1)  # [c,1,h,w]
            frame_tensor1 = repeat(image_tensor1, 'c t h w -> c (repeat t) h w', repeat=video_frames // 2)
            frame_tensor2 = repeat(image_tensor2, 'c t h w -> c (repeat t) h w', repeat=video_frames // 2)
            frame_tensor = torch.cat([frame_tensor1, frame_tensor2], dim=1)
            _, filename = os.path.split(file_list[idx * 2])
        else:
            image = Image.open(file_list[idx]).convert('RGB')
            image_tensor = transform(image).unsqueeze(1)  # [c,1,h,w]
            frame_tensor = repeat(image_tensor, 'c t h w -> c (repeat t) h w', repeat=video_frames)
            _, filename = os.path.split(file_list[idx])

        data_list.append(frame_tensor)
        filename_list.append(filename)

    return filename_list, data_list, prompt_list


def image_guided_synthesis(model, prompts, videos, noise_shape, n_samples=1, ddim_steps=50, ddim_eta=1., \
                           unconditional_guidance_scale=1.0, cfg_img=None, fs=None, text_input=False,
                           multiple_cond_cfg=False, loop=False, interp=False, timestep_spacing='uniform',
                           guidance_rescale=0.0, **kwargs):
    ddim_sampler = DDIMSampler(model) if not multiple_cond_cfg else DDIMSampler_multicond(model)
    batch_size = noise_shape[0]
    fs = torch.tensor([fs] * batch_size, dtype=torch.long, device=model.device)

    if not text_input:
        prompts = [""] * batch_size

    img = videos[:, :, 0]  # bchw
    img_emb = model.embedder(img)  ## blc
    img_emb = model.image_proj_model(img_emb)

    cond_emb = model.get_learned_conditioning(prompts)
    cond = {"c_crossattn": [torch.cat([cond_emb, img_emb], dim=1)]}
    if model.model.conditioning_key == 'hybrid':
        z = get_latent_z(model, videos)  # b c t h w
        if loop or interp:
            img_cat_cond = torch.zeros_like(z)
            img_cat_cond[:, :, 0, :, :] = z[:, :, 0, :, :]
            img_cat_cond[:, :, -1, :, :] = z[:, :, -1, :, :]
        else:
            img_cat_cond = z[:, :, :1, :, :]
            img_cat_cond = repeat(img_cat_cond, 'b c t h w -> b c (repeat t) h w', repeat=z.shape[2])
        cond["c_concat"] = [img_cat_cond]  # b c 1 h w

    if unconditional_guidance_scale != 1.0:
        if model.uncond_type == "empty_seq":
            prompts = batch_size * [""]
            uc_emb = model.get_learned_conditioning(prompts)
        elif model.uncond_type == "zero_embed":
            uc_emb = torch.zeros_like(cond_emb)
        uc_img_emb = model.embedder(torch.zeros_like(img))  ## b l c
        uc_img_emb = model.image_proj_model(uc_img_emb)
        uc = {"c_crossattn": [torch.cat([uc_emb, uc_img_emb], dim=1)]}
        if model.model.conditioning_key == 'hybrid':
            uc["c_concat"] = [img_cat_cond]
    else:
        uc = None

    ## we need one more unconditioning image=yes, text=""
    if multiple_cond_cfg and cfg_img != 1.0:
        uc_2 = {"c_crossattn": [torch.cat([uc_emb, img_emb], dim=1)]}
        if model.model.conditioning_key == 'hybrid':
            uc_2["c_concat"] = [img_cat_cond]
        kwargs.update({"unconditional_conditioning_img_nonetext": uc_2})
    else:
        kwargs.update({"unconditional_conditioning_img_nonetext": None})

    z0 = None
    cond_mask = None

    batch_variants = []
    for _ in range(n_samples):

        if z0 is not None:
            cond_z0 = z0.clone()
            kwargs.update({"clean_cond": True})
        else:
            cond_z0 = None
        if ddim_sampler is not None:
            samples, _ = ddim_sampler.sample(S=ddim_steps,
                                             conditioning=cond,
                                             batch_size=batch_size,
                                             shape=noise_shape[1:],
                                             verbose=False,
                                             unconditional_guidance_scale=unconditional_guidance_scale,
                                             unconditional_conditioning=uc,
                                             eta=ddim_eta,
                                             cfg_img=cfg_img,
                                             mask=cond_mask,
                                             x0=cond_z0,
                                             fs=fs,
                                             timestep_spacing=timestep_spacing,
                                             guidance_rescale=guidance_rescale,
                                             **kwargs
                                             )

        ## reconstruct from latent to pixel space
        batch_images = model.decode_first_stage(samples)
        batch_variants.append(batch_images)
    ## variants, batch, c, t, h, w
    batch_variants = torch.stack(batch_variants)
    return batch_variants.permute(1, 0, 2, 3, 4, 5)


def save_results_seperate(prompt, samples, filename, fakedir, fps=10, loop=False):
    prompt = prompt[0] if isinstance(prompt, list) else prompt

    ## save video
    videos = [samples]
    savedirs = [fakedir]
    for idx, video in enumerate(videos):
        if video is None:
            continue
        # b,c,t,h,w
        video = video.detach().cpu()
        if loop: # remove the last frame
            video = video[:,:,:-1,...]
        video = torch.clamp(video.float(), -1., 1.)
        n = video.shape[0]
        for i in range(n):
            grid = video[i,...]
            grid = (grid + 1.0) / 2.0
            grid = (grid * 255).to(torch.uint8).permute(1, 2, 3, 0) #thwc
            path = os.path.join(savedirs[idx].replace('samples', 'samples_separate'), f'{filename.split(".")[0]}_sample{i}.mp4')
            torchvision.io.write_video(path, grid, fps=fps, video_codec='h264', options={'crf': '10'})

def save_results_frame(out_dir, samples, filename, run_on_whole_clip=False):
    """
    Save the results by frame
    """
    if not os.path.exists(out_dir):
        print("Output directory doesn't exist, creating it")
        os.makedirs(out_dir)
    samples = samples.detach().cpu()
    samples = torch.clamp(samples.float(), -1., 1.)
    sample = samples[0]  # Suppose batch size is 1 in the inference
    sample = (sample + 1.0) / 2.0
    sample = (sample * 255).to(torch.uint8).permute(1, 2, 3, 0)
    sample = sample.numpy()
    n_frames = sample.shape[0]
    if not run_on_whole_clip:
        for i in range(n_frames):
            frame = sample[i, ...]
            img = Image.fromarray(frame)
            img.save(os.path.join(out_dir, f'{filename.split(".")[0]}_{i}.png'))
    else:
        ## Only save the interpolated frames between first two frames
        ## frame 1,2,3 should be the results, no matter what interp type is
        frame_0 = sample[0, ...]
        frame_1 = sample[1, ...]
        frame_2 = sample[2, ...]
        frame_3 = sample[3, ...]
        img_0 = Image.fromarray(frame_0)
        img_1 = Image.fromarray(frame_1)
        img_2 = Image.fromarray(frame_2)
        img_3 = Image.fromarray(frame_3)
        img_0.save(os.path.join(out_dir, f'{filename.split(".")[0]}_0.png'))
        img_1.save(os.path.join(out_dir, f'{filename.split(".")[0]}_1.png'))
        img_2.save(os.path.join(out_dir, f'{filename.split(".")[0]}_2.png'))
        img_3.save(os.path.join(out_dir, f'{filename.split(".")[0]}_3.png'))

def run_frm_interp(args):
    """
    Run frame interpolation using DynamiCrafter.
    """
    print('Running frame interpolation.')
    gpu_num = 1
    gpu_no = 0
    cfg_path = args.cfg_path
    cfg = OmegaConf.load(cfg_path)
    model_cfg = cfg.pop("model", OmegaConf.create())

    model_cfg['params']['unet_config']['params']['use_checkpoint'] = False
    model = instantiate_from_config(model_cfg)
    ## Model profiling
    model_profiling = False
    if model_profiling:
        submodel_openclip_embedder = model.cond_stage_model
        submodel_openclip_image_embedder_v2 = model.embedder
        submodel_autoencoder_kl = model.first_stage_model
        submodel_resampler = model.image_proj_model
        submodel_unet = model.model
        submodel_list = [submodel_openclip_embedder, submodel_openclip_image_embedder_v2, submodel_autoencoder_kl, submodel_resampler, submodel_unet]
        for submodel in submodel_list:
            model_description, model_no_params = get_network_description(submodel)
            print(f"Network description: {model_description}, Num params: {model_no_params}")
    ###
    model = model.cuda(gpu_no)
    model.perframe_ae = args.perframe_ae
    assert os.path.exists(args.ckpt_path), "Error: checkpoint Not Found!"
    model = load_model_checkpoint(model, args.ckpt_path)
    model.eval()

    ## run over data
    assert (args.height % 16 == 0) and (args.width % 16 == 0), "Error: image size [h,w] should be multiples of 16!"
    assert args.bs == 1, "Current implementation only support [batch size = 1]!"
    ## latent noise shape
    h, w = args.height // 8, args.width // 8
    channels = model.model.diffusion_model.out_channels
    n_frames = args.video_length
    print(f'Inference with {n_frames} frames')
    noise_shape = [args.bs, channels, n_frames, h, w]

    # fakedir = os.path.join(args.savedir, "samples")
    # fakedir_separate = os.path.join(args.savedir, "samples_separate")
    #
    # # os.makedirs(fakedir, exist_ok=True)
    # os.makedirs(fakedir_separate, exist_ok=True)

    frame_tensor = load_frames(frame_path_list=args.frame_path_list,
                               height=args.height,
                               width=args.width,
                               n_frames=n_frames)
    data_list = [frame_tensor]
    prompt = load_prompt(args.prompt_file_path)
    prompt_list = [prompt]
    _, filename = os.path.split(args.frame_path_list[0])
    filename_list = [filename]

    ## prompt file setting
    # assert os.path.exists(args.prompt_dir), "Error: prompt file Not Found!"
    # filename_list, data_list, prompt_list = load_data_prompts(args.prompt_dir, video_size=(args.height, args.width),
    #                                                           video_frames=n_frames, interp=args.interp)
    num_samples = len(prompt_list)
    samples_split = num_samples // gpu_num
    print('Prompts testing [rank:%d] %d/%d samples loaded.' % (gpu_no, samples_split, num_samples))
    # indices = random.choices(list(range(0, num_samples)), k=samples_per_device)
    indices = list(range(samples_split * gpu_no, samples_split * (gpu_no + 1)))
    prompt_list_rank = [prompt_list[i] for i in indices]
    data_list_rank = [data_list[i] for i in indices]
    filename_list_rank = [filename_list[i] for i in indices]

    # make out_dir
    out_dir = args.out_dir
    out_dir = f"{out_dir}_n_frames_{args.video_length}_fs_{args.frame_stride}_ddim_steps_{args.ddim_steps}_seed_{args.seed}"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    start = time.time()
    with torch.no_grad(), torch.cuda.amp.autocast():
        for idx, indice in tqdm(enumerate(range(0, len(prompt_list_rank), args.bs)), desc='Sample Batch'):
            prompts = prompt_list_rank[indice:indice + args.bs]
            videos = data_list_rank[indice:indice + args.bs]
            filenames = filename_list_rank[indice:indice + args.bs]
            if isinstance(videos, list):
                videos = torch.stack(videos, dim=0).to("cuda")
            else:
                videos = videos.unsqueeze(0).to("cuda")

            batch_samples = image_guided_synthesis(model, prompts, videos, noise_shape, args.n_samples, args.ddim_steps,
                                                   args.ddim_eta,
                                                   args.unconditional_guidance_scale, args.cfg_img, args.frame_stride,
                                                   args.text_input, args.multiple_cond_cfg, args.loop, args.interp,
                                                   args.timestep_spacing, args.guidance_rescale)

            ## save each example individually
            for nn, samples in enumerate(batch_samples):
                ## samples : [n_samples,c,t,h,w]
                prompt = prompts[nn]
                filename = filenames[nn]
                # save_results(prompt, samples, filename, fakedir, fps=8, loop=args.loop)
                save_results_frame(out_dir=out_dir,
                                   samples=samples,
                                   filename=filename)
                # save_results_seperate(prompt, samples, filename, fakedir, fps=8, loop=args.loop)

    print(f"Saved in {out_dir}. Time used: {(time.time() - start):.2f} seconds")


def run_clip_interp(args):
    """
    Run the model on the whole clip
    In the args, run_on_whole_clip should be set to true and in_dir should be set.
    """
    ## Get the frame list
    img_file_name_list = os.listdir(args.in_dir)
    input_frame_name_list = []
    for img_file_name in img_file_name_list:
        if img_file_name.endswith(".png"):
            input_frame_name_list.append(img_file_name)  # Input assumed to be png
    input_frame_name_list = sorted(input_frame_name_list)  # Sort the input frame names
    total_frame_num = len(input_frame_name_list)

    in_frame_num = args.in_frame_num
    assert total_frame_num >= in_frame_num

    n_iters = total_frame_num - in_frame_num + 1

    print('Running frame interpolation on the whole test clip.')
    gpu_num = 1
    gpu_no = 0
    cfg_path = args.cfg_path
    cfg = OmegaConf.load(cfg_path)
    model_cfg = cfg.pop("model", OmegaConf.create())

    model_cfg['params']['unet_config']['params']['use_checkpoint'] = False
    model = instantiate_from_config(model_cfg)

    model = model.cuda(gpu_no)
    model.perframe_ae = args.perframe_ae
    assert os.path.exists(args.ckpt_path), "Error: checkpoint Not Found!"
    model = load_model_checkpoint(model, args.ckpt_path)
    model.eval()

    ## run over data
    assert (args.height % 16 == 0) and (args.width % 16 == 0), "Error: image size [h,w] should be multiples of 16!"
    assert args.bs == 1, "Current implementation only support [batch size = 1]!"
    ## latent noise shape
    h, w = args.height // 8, args.width // 8
    channels = model.model.diffusion_model.out_channels
    n_frames = args.video_length
    print(f'Inference with {n_frames} frames')
    noise_shape = [args.bs, channels, n_frames, h, w]

    start = time.time()
    for w_i in range(n_iters):
        frame_name_list = input_frame_name_list[w_i:w_i+in_frame_num]
        frame_path_list = [os.path.join(args.in_dir, f_name) for f_name in frame_name_list]
        frame_path_list = sorted(frame_path_list)  # Sort again
        frame_tensor = load_frames(frame_path_list=frame_path_list,
                                height=args.height,
                                width=args.width,
                                n_frames=n_frames,
                                interp_type="5in_16out")
        data_list = [frame_tensor]
        prompt = load_prompt(args.prompt_file_path)
        prompt = ""
        prompt_list = [prompt]
        _, filename = os.path.split(frame_path_list[0])
        filename_list = [filename]

        num_samples = len(prompt_list)
        samples_split = num_samples // gpu_num
        print('Prompts testing [rank:%d] %d/%d samples loaded.' % (gpu_no, samples_split, num_samples))
        # indices = random.choices(list(range(0, num_samples)), k=samples_per_device)
        indices = list(range(samples_split * gpu_no, samples_split * (gpu_no + 1)))
        prompt_list_rank = [prompt_list[i] for i in indices]
        data_list_rank = [data_list[i] for i in indices]
        filename_list_rank = [filename_list[i] for i in indices]

        # make out_dir
        out_dir = args.out_dir
        out_dir = f"{out_dir}_n_frames_{args.video_length}_fs_{args.frame_stride}_ddim_steps_{args.ddim_steps}_seed_{args.seed}"
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        with torch.no_grad(), torch.cuda.amp.autocast():
            for idx, indice in tqdm(enumerate(range(0, len(prompt_list_rank), args.bs)), desc='Sample Batch'):
                prompts = prompt_list_rank[indice:indice + args.bs]
                videos = data_list_rank[indice:indice + args.bs]
                filenames = filename_list_rank[indice:indice + args.bs]
                if isinstance(videos, list):
                    videos = torch.stack(videos, dim=0).to("cuda")
                else:
                    videos = videos.unsqueeze(0).to("cuda")

                batch_samples = image_guided_synthesis(model, prompts, videos, noise_shape, args.n_samples,
                                                       args.ddim_steps,
                                                       args.ddim_eta,
                                                       args.unconditional_guidance_scale, args.cfg_img,
                                                       args.frame_stride,
                                                       args.text_input, args.multiple_cond_cfg, args.loop, args.interp,
                                                       args.timestep_spacing, args.guidance_rescale)

                ## save each example individually
                for nn, samples in enumerate(batch_samples):
                    ## samples : [n_samples,c,t,h,w]
                    prompt = prompts[nn]
                    filename = filenames[nn]
                    # save_results(prompt, samples, filename, fakedir, fps=8, loop=args.loop)
                    save_results_frame(out_dir=out_dir,
                                       samples=samples,
                                       filename=filename,
                                       run_on_whole_clip=True)
                    # save_results_seperate(prompt, samples, filename, fakedir, fps=8, loop=args.loop)
    print(f"Whole clip interpolated. Time used: {(time.time() - start):.2f} seconds")


def main():
    args = parse_args()
    task_type = args.task

    seed = args.seed
    seed_everything(seed)

    if task_type == 'interp':
        if not args.run_on_whole_clip:
            run_frm_interp(args=args)
        else:
            run_clip_interp(args=args)
    else:
        raise NotImplementedError('Task type {} not implemented'.format(task_type))


if __name__ == "__main__":
    main()