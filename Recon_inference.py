import os
import sys
import json
import argparse
import numpy as np
import math
from einops import rearrange
import time
import random
import string
import h5py
from tqdm import tqdm
import webdataset as wds
from contextlib import redirect_stdout
from functools import partial

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision import transforms
from accelerate import Accelerator
from transformers import CLIPImageProcessor
from diffusers import AutoPipelineForText2Image
from pipeline_stable_diffusion_xl import StableDiffusionXLPipelineRouting

# SDXL unCLIP requires code from https://github.com/Stability-AI/generative-models/tree/main
sys.path.append('MindEyeV2/src/generative_models/')
sys.path.append('MindEyeV2/src')
import sgm
from MindEyeV2.src.generative_models.sgm.models.diffusion import DiffusionEngine
from MindEyeV2.src.generative_models.sgm.util import append_dims
from omegaconf import OmegaConf

# tf32 data type is faster than standard float32
torch.backends.cuda.matmul.allow_tf32 = True

# custom functions #
import MindEyeV2.src.utils as utils
from MindEyeV2.src.models import *
from clip_encoders import CLIPImageEncoder
from brain_encoder import BrainEncoder, BrainDiffusionPriorEncoder
from brain_moe_encoder import BrainMoE, BrainMoEMulti, BrainMoEHier
from load_data import load_nsd, load_train_data
from routers import *


accelerator = Accelerator(split_batches=False, mixed_precision="fp16")
device = accelerator.device
world_size = accelerator.state.num_processes
distributed = not accelerator.state.distributed_type == 'NO'
print("device:",device)

diffuse_router_image = None
diffuse_router_text = None

image_processor = CLIPImageProcessor.from_pretrained("laion/CLIP-ViT-bigG-14-laion2B-39B-b160k", cache_dir="data/cache")
pipe_tmp = StableDiffusionXLPipelineRouting.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True, cache_dir='data/cache')
tokenizer_2, text_enc_2 = pipe_tmp.tokenizer_2, pipe_tmp.text_encoder_2
text_ids = tokenizer_2('a cat on a shelf', return_tensors="pt", padding="max_length", truncation=True, max_length=tokenizer_2.model_max_length)
sos_embed = text_enc_2(text_ids.input_ids, output_hidden_states=True).hidden_states[-2][:, :1]
tokenizer_1, text_enc_1 = pipe_tmp.tokenizer, pipe_tmp.text_encoder
text_ids = tokenizer_2('a cat on a shelf', return_tensors="pt", padding="max_length", truncation=True, max_length=tokenizer_1.model_max_length)
sos_embed_L = text_enc_1(text_ids.input_ids, output_hidden_states=True).hidden_states[-1][:, :1]

empty_embed = text_enc_2(tokenizer_2('', return_tensors="pt", padding="max_length", truncation=True, max_length=tokenizer_2.model_max_length).input_ids, output_hidden_states=True).hidden_states[-2]

@torch.no_grad()
def get_image_embed(model, diffusion_prior, voxel_list, subj_list, accelerator):
    if 'MoE' in model_name_image:
        backbones, clip_voxels, _, _ = model(voxel_list, subj_list, training=False, return_exp_out=True if lora_ckpt else False)
    else:
        backbones, clip_voxels, _ = model(voxel_list, subj_list)
    return backbones, clip_voxels

@torch.no_grad()
def prior_image(diffusion_prior, backbones):
    prior_out = diffusion_prior.prior.p_sample_loop(backbones.shape, text_cond=dict(text_embed=backbones), cond_scale=1.5, timesteps=30)
    return prior_out

@torch.no_grad()
def get_text_embed(model, diffusion_prior, voxel_list, subj_list, accelerator, text_L=False):
    if 'MoE' in model_name_text:
        if text_L:
            backbones, clip_voxels, _, _ = model(voxel_list, subj_list, training=False, return_exp_out=False)    
        backbones, clip_voxels, _, _ = model(voxel_list, subj_list, training=False, return_exp_out=True if lora_ckpt else False)
    else:
        backbones, clip_voxels, _ = model(voxel_list, subj_list)
    
    if isinstance(backbones, list):
        bz, seq_len = backbones[0].shape[1], backbones[0].shape[-1]
    else:
        bz, seq_len = backbones.shape[0], backbones.shape[-1]
    
    pooled_embed = torch.zeros(bz, seq_len).to(accelerator.device)
    return backbones, clip_voxels, pooled_embed
   
@torch.no_grad()
def prior_text(diffusion_prior, backbones):
    prior_out = diffusion_prior.prior.p_sample_loop(backbones.shape, text_cond=dict(text_embed=backbones), cond_scale=1.5, timesteps=30)
    if prior_out.shape[-1] == 768:
        prior_out = torch.cat([sos_embed_L.repeat(prior_out.shape[0], 1, 1).to(accelerator.device), prior_out.repeat(1, 76, 1)], dim=1)

    return prior_out


def check_resume_progress(subj, project_name, model_name, device):
    """Check if there are files to resume from and return progress info"""
    train_progress_file = f"evals/{subj}/{project_name}/{model_name}/train_progress_{device}.json"
    val_progress_file = f"evals/{subj}/{project_name}/{model_name}/val_progress_{device}.json"
    
    train_progress = {"completed": False, "current_step": 0}
    val_progress = {"completed": False, "current_batch": 0}
    
    # Check training progress
    if os.path.exists(train_progress_file):
        with open(train_progress_file, 'r') as f:
            train_progress = json.load(f)
    
    # Check validation progress
    if os.path.exists(val_progress_file):
        with open(val_progress_file, 'r') as f:
            val_progress = json.load(f)
            
    return train_progress, val_progress


def save_checkpoint(subj, project_name, model_name, device, phase, progress_info, 
                   indices, reconstructions, save_full=False, clip_img=None, clip_txt=None):
    """Save checkpoint with current progress"""
    if saving_dir:
        checkpoint_dir = f"evals/{subj}/{project_name}/{saving_dir}"
    else:
        checkpoint_dir = f"evals/{subj}/{project_name}/{model_name}"
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Save progress info
    progress_file = f"{checkpoint_dir}/{phase}_progress_{device}.json"
    with open(progress_file, 'w') as f:
        json.dump(progress_info, f)
    
    # Save indices
    if phase == "train":
        np.save(f"{checkpoint_dir}/image_indices_train_temp_{device}.npy", np.concatenate(indices))
        torch.save(reconstructions, f"{checkpoint_dir}/all_recons_train_temp_{device}.pt")
        if clip_img: np.save(f"{checkpoint_dir}/clip_img_train_temp_{device}.npy", np.concatenate(clip_img, 0))
        if clip_txt: np.save(f"{checkpoint_dir}/clip_txt_train_temp_{device}.npy", np.concatenate(clip_txt, 0))

        # Save reconstructions periodically (every chunk) or at completion
        if save_full or progress_info["completed"]:
            # Save final results
            np.save(f"{checkpoint_dir}/image_indices_train_{device}.npy", np.concatenate(indices))
            torch.save(reconstructions, f"{checkpoint_dir}/all_recons_train_{device}.pt")
            if clip_img: np.save(f"{checkpoint_dir}/clip_img_train_{device}.npy", np.concatenate(clip_img, 0))
            if clip_txt: np.save(f"{checkpoint_dir}/clip_txt_train_{device}.npy", np.concatenate(clip_txt, 0))
    else:  # validation
        if save_full or progress_info["completed"]:
            # Save final results
            np.save(f"{checkpoint_dir}/image_indices_val_{device}.npy", np.concatenate(indices))
            torch.save(reconstructions, f"{checkpoint_dir}/all_recons_val_{device}.pt")

            if clip_img: np.save(f"{checkpoint_dir}/clip_img_val_{device}.npy", np.concatenate(clip_img, 0))
            if clip_txt: np.save(f"{checkpoint_dir}/clip_txt_val_{device}.npy", np.concatenate(clip_txt, 0))


parser = argparse.ArgumentParser(description="Model Training Configuration")
parser.add_argument(
    "--model_name", type=str, default="testing",
    help="will load ckpt for model found in ../train_logs/model_name",
)
parser.add_argument(
    "--data_path", type=str, default='data/NSD',
    help="Path to where NSD data is stored / where to download it to",
)
parser.add_argument(
    "--cache_dir", type=str, default='data/cache',
    help="Path to where misc. files downloaded from huggingface are stored. Defaults to current src directory.",
)
parser.add_argument(
    "--subj",type=int, default=1, choices=[1,2,3,4,5,6,7,8],
    help="Validate on which subject?",
)
parser.add_argument(
    "--blurry_recon",action=argparse.BooleanOptionalAction,default=False,
)
parser.add_argument(
    "--n_blocks",type=int,default=2,
)
parser.add_argument(
    "--hidden_dim",type=int,default=1024,
)
parser.add_argument(
    "--new_test",action=argparse.BooleanOptionalAction,default=True,
)
parser.add_argument(
    "--seed",type=int,default=42,
)
parser.add_argument(
    "--batch_size",type=int, default=24,
)
parser.add_argument(
    "--num_sessions",type=int,default=40,
)
parser.add_argument(
    "--model_name_image", type=str, default='single_subj1_40sess_ViT-BigG-img-prior-noblur-4096-BrainMoEMulti2-2L3F2--all+',
    help="name of model, used for ckpt saving and wandb logging (if enabled)",
)
parser.add_argument(
    '--model_name_text', type=str, default='single_subj1_40sess_ViT_-cap5-prior-noblur-4096-BigG-BrainMoEMulti2-2L3F2--all+',
)
parser.add_argument(
    '--model_name_text_L', type=str, default='', help='additional ViT-L text encoder'
)
parser.add_argument(
    '--project_name', type=str, default='Image',
)
parser.add_argument(
    '--saving_dir', type=str, default='test',
)
parser.add_argument(
    '--lora_ckpt', type=str, default='MoEMulti2-noMeta-SpaceRouterCrossAttn-TimeRouterAttn_temp1.5_soft--all+',
)
parser.add_argument(
    '--multi_subject', action=argparse.BooleanOptionalAction, default=False,
)
parser.add_argument(
    '--b_size', type=int, default=-1,
)

if utils.is_interactive():
    args = parser.parse_args(jupyter_args)
else:
    args = parser.parse_args()
num_devices = torch.cuda.device_count()

# create global variables without the args prefix
for attribute_name in vars(args).keys():
    globals()[attribute_name] = getattr(args, attribute_name)
    
use_prior = True
model_name = model_name_image if model_name_image else ""
model_name += f"---{model_name_text}" if model_name_text else ""
if model_name_text and not model_name_image:  model_name = model_name_text
subj_list = [subj]

# seed all random functions
utils.seed_everything(seed)

with open(f'{data_path}/coco_captions_keywords_5.json', 'r') as f:
    descs = json.load(f)

pipe = StableDiffusionXLPipelineRouting.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    variant="fp16",
    cache_dir='data/cache',
).to("cuda")

# pipe.load_lora_weights(f'checkpoints/sdxl-finetuned-lora/MoEMulti2-noMeta-trained_SpaceRouterCrossAttn--all/best_model', weight_name="pytorch_lora_weights.safetensors")
if lora_ckpt:
    pipe.load_lora_weights(f'checkpoints/sdxl-finetuned-lora/{lora_ckpt}/best_model', weight_name="pytorch_lora_weights.safetensors")
    # pipe.unet.load_lora_adapter()

    diffuse_router_image = DiffuseRouter(
        time_router_class=DiffuseTimeRouterAttn,
        space_router_class=DiffuseSpaceRouterCrossAttn,
        num_granularity_levels=4,
        num_experts_per_granularity=[2, 4, 8, 16],
    )
    diffuse_router_image_ckpt = torch.load(f'checkpoints/sdxl-finetuned-lora/{lora_ckpt}/best_model/best_diffuse_router_image.pth', map_location='cpu')
    diffuse_router_image.load_state_dict(diffuse_router_image_ckpt)
    diffuse_router_image.requires_grad_(False)
    diffuse_router_image.to(device).eval()

    diffuse_router_text = DiffuseRouter(
        time_router_class=DiffuseTimeRouterAttn,
        space_router_class=DiffuseSpaceRouterCrossAttn,
        num_granularity_levels=4,
        num_experts_per_granularity=[2, 4, 8, 16],
    )
    diffuse_router_text_ckpt = torch.load(f'checkpoints/sdxl-finetuned-lora/{lora_ckpt}/best_model/best_diffuse_router_text.pth', map_location='cpu')
    diffuse_router_text.load_state_dict(diffuse_router_text_ckpt)
    diffuse_router_text.requires_grad_(False)
    diffuse_router_text.to(device).eval()  
    

pipe._progress_bar_config = {'disable': True}

train_dls, voxels, batch_size, test_dataloader, num_voxels_list, subj_list, num_iterations_per_epoch = load_nsd(num_sessions, subj, num_devices, batch_size, batch_size, args.multi_subject)
train_dls = train_dls.values()
test_dataloader, *train_dls = accelerator.prepare(test_dataloader, *train_dls)


if 'Image' in project_name:
    pipe.load_ip_adapter(
        "h94/IP-Adapter",
        subfolder="sdxl_models",
        weight_name="ip-adapter_sdxl.bin",  # or "ip-adapter-plus_sdxl_vit-h.safetensors"
        cache_dir='data/cache',
    )
    pipe.set_ip_adapter_scale(1.0) 


arch = "ViT-bigG-14" if 'BigG' in model_name else "ViT-L-14"
version = "laion2b_s39b_b160k" if arch == "ViT-bigG-14" else "laion2b_s32b_b82k"

clip_embedder_image = CLIPImageEncoder('BigG', is_proj=True, return_type='pooled')
clip_embedder_image.to(device)
clip_emb_dim = 1280
clip_seq_dim = 1


if 'Image' in project_name:
    clip_emb_dim, clip_seq_dim = 1280, 1
    # brain_model_image = BrainEncoder(num_voxels_list, 4096, blurry_recon, 4, clip_emb_dim, clip_seq_dim, 1, interm_out=False, enc_version='v1')
    brain_model_image = BrainMoEMulti(num_voxels_list, 4096, blurry_recon, 4, clip_emb_dim, clip_seq_dim, 1, interm_out=False, enc_version='v1', num_exp_0=2, capacity_factor_0=1, num_exp_layer=3, exp_factor_list=[2, 2, 2], cap_fac_list=[1, 1, 1], meta=False, b_size=args.b_size)
    diffuse_prior_image = BrainDiffusionPriorEncoder(clip_emb_dim, depth=6, dim_head=52, heads=clip_emb_dim//52, clip_seq_dim=clip_seq_dim, timesteps=100)
if 'Text' in project_name:
    hidden_dim, clip_emb_dim, clip_seq_dim = 4096, 1280, 1
    # brain_model_text = BrainEncoder(num_voxels_list, hidden_dim, blurry_recon, 4, clip_emb_dim, clip_seq_dim, 1, interm_out=False, enc_version='v1')
    brain_model_text = BrainMoEMulti(num_voxels_list, 4096, blurry_recon, 4, clip_emb_dim, clip_seq_dim, 1, interm_out=False, enc_version='v1', num_exp_0=2, capacity_factor_0=1, num_exp_layer=3, exp_factor_list=[2, 2, 2], cap_fac_list=[1, 1, 1], meta=False, b_size=args.b_size)
    diffuse_prior_text = BrainDiffusionPriorEncoder(clip_emb_dim, depth=6, dim_head=52, heads=clip_emb_dim//52, clip_seq_dim=clip_seq_dim, timesteps=100)
if 'TextL' in project_name:
    clip_emb_dim, clip_seq_dim = 768, 1
    brain_model_text_L = BrainEncoder(num_voxels_list, hidden_dim, blurry_recon, 4, clip_emb_dim, clip_seq_dim, 1, interm_out=False, enc_version='v1')
    diffuse_prior_text_L = BrainDiffusionPriorEncoder(clip_emb_dim, depth=6, dim_head=52, heads=clip_emb_dim//52, clip_seq_dim=clip_seq_dim, timesteps=100)


if 'Image' in project_name:
    ckpt_image = torch.load(f'checkpoints/{args.model_name_image}/best_mse.pth', map_location='cpu')
    brain_model_image.load_state_dict(ckpt_image['model_state_dict'])
    diffuse_prior_image.load_state_dict(ckpt_image['prior_state_dict'])
    brain_model_image.requires_grad_(False)
    diffuse_prior_image.requires_grad_(False)
    brain_model_image.to(device).eval()
    diffuse_prior_image.to(device).eval()
    del ckpt_image
if 'Text' in project_name:
    ckpt_text = torch.load(f'checkpoints/{args.model_name_text}/best_mse.pth', map_location='cpu')
    brain_model_text.load_state_dict(ckpt_text['model_state_dict'])
    diffuse_prior_text.load_state_dict(ckpt_text['prior_state_dict'])
    brain_model_text.requires_grad_(False)
    diffuse_prior_text.requires_grad_(False)
    brain_model_text.to(device).eval()
    diffuse_prior_text.to(device).eval()
    del ckpt_text
# if 'TextL' in project_name:
#     ckpt_textL = torch.load(f'checkpoints/{args.model_name_text_L}/best_mse.pth', map_location='cpu')
#     brain_model_text_L.load_state_dict(ckpt_textL['model_state_dict'])
#     diffuse_prior_text_L.load_state_dict(ckpt_textL['prior_state_dict'])
#     del ckpt_textL
#     brain_model_text_L.requires_grad_(False)
#     diffuse_prior_text_L.requires_grad_(False)
#     brain_model_text_L.to(device).eval()
#     diffuse_prior_text_L.to(device).eval()


all_recons_train, all_recons_val = [], []
all_clip_img_train, all_clip_img_val = [], []
all_clip_txt_train, all_clip_txt_val = [], []
image_indices_train, image_indices_val = [], []
checkpoint_freq = 5
assert lora_ckpt, "Always using a router for this script"

train_progress, val_progress = check_resume_progress(subj, project_name + '_', model_name, device)

if not val_progress["completed"]:
    # If resuming, try to load existing partial results
    if val_progress["current_batch"] > 0:
        try:
            temp_indices_file = f"evals/{subj}/{project_name}/{model_name}/image_indices_val_temp_{device}.npy"
            recons_file = f"evals/{subj}/{project_name}/{model_name}/all_recons_val_{device}.pt"

            if os.path.exists(temp_indices_file) and os.path.exists(recons_file):
                image_indices_val = [np.load(temp_indices_file)]
                all_recons_val = [torch.load(recons_file)]
                print(f"Resuming validation from batch {val_progress['current_batch']}")

            if 'Image' in project_name:
                temp_clip_img_file = f"evals/{subj}/{project_name}/{model_name}/clip_img_val_temp_{device}.npy"
                if os.path.exists(temp_clip_img_file):
                    all_clip_img_val = [np.load(temp_clip_img_file)]
            if 'Text' in project_name:
                temp_clip_txt_file = f"evals/{subj}/{project_name}/{model_name}/clip_txt_val_temp_{device}.npy"
                if os.path.exists(temp_clip_txt_file):
                    all_clip_txt_val = [np.load(temp_clip_txt_file)]

        except Exception as e:
            print(f"Error loading previous validation results: {e}")
            val_progress["current_batch"] = 0
            all_recons_val, image_indices_val = [], []

    total_val_batches = len(test_dataloader)

    for batch_idx, batch in enumerate(tqdm(test_dataloader, desc="Validation", disable=not accelerator.is_main_process)):
        if batch_idx < val_progress["current_batch"]:
            continue

        pixel_values, voxel_values = batch['image'], batch['voxel'].to(accelerator.device)
        image_indices_val.append(batch['img_idx'].cpu().numpy())
        text = [descs[str(i)] for i in image_indices_val[-1]]
        text = [d[0] for d in text]
        
        prompts_img, prompts_txt = [], []
        prompts_txt_L = []
        clip_imgs, clip_txts = [], []
        with accelerator.autocast():
            for rep in range(3):
                if 'Image' in project_name:
                    prompt_embed_image, clip_img = get_image_embed(brain_model_image, diffuse_prior_image, [voxel_values[:, rep]], subj_list[:1], accelerator)
                    
                    prompts_img.append(prompt_embed_image)
                    clip_imgs.append(clip_img.cpu())

                if 'Text' in project_name:
                    prompt_embed_text, clip_txt, pooled_embed = get_text_embed(brain_model_text, diffuse_prior_text, [voxel_values[:, rep]], subj_list[:1], accelerator)
                    clip_txts.append(clip_txt.cpu())
                    
                    if lora_ckpt:  # do this inside SDXL pipeline
                        prompts_txt.append(prompt_embed_text)
                        if 'TextL' in project_name:
                            # prompt_embed_text_L, _, _ = get_text_embed(brain_model_text_L, diffuse_prior_text_L, [voxel_values[:, rep]], subj_list, accelerator)
                            prompt_embed_text_L = tokenizer_1(text, return_tensors="pt", padding="max_length", truncation=True, max_length=tokenizer_2.model_max_length)
                            prompt_embed_text_L = text_enc_1(prompt_embed_text_L.input_ids, output_hidden_states=True).hidden_states[-2]
                            prompts_txt_L.append(prompt_embed_text_L)
                    else:
                        raise Exception

            if len(prompts_img):
                if isinstance(prompts_img[0], list):
                    prompt_embed_image = [torch.zeros_like(prompts_img[0][i]) for i in range(len(prompts_img[0]))]
                    for elem in prompts_img:
                        for i in range(len(elem)):
                            prompt_embed_image[i] += elem[i]
                    prompt_embed_image = [p / len(prompts_img) for p in prompt_embed_image]
                else:
                    prompt_embed_image = torch.stack(prompts_img, 0).mean(0)
                clip_imgs = torch.stack(clip_imgs, 0).mean(0)

            if len(prompts_txt):
                if isinstance(prompts_txt[0], list):
                    prompt_embed_text = [torch.zeros_like(prompts_txt[0][i]) for i in range(len(prompts_txt[0]))]
                    for elem in prompts_txt:
                        for i in range(len(elem)):
                            prompt_embed_text[i] += elem[i]
                    prompt_embed_text = [p / len(prompts_txt) for p in prompt_embed_text]
                else:
                    prompt_embed_text = torch.stack(prompts_txt, 0).mean(0)
                clip_txts = torch.stack(clip_txts, 0).mean(0)
            
            if len(prompts_txt_L):
                prompt_embed_text_L = torch.stack(prompts_txt_L, 0).mean(0).to(accelerator.device)
            else:
                prompt_embed_text_L = None

        with torch.amp.autocast(dtype=torch.float32, device_type='cuda'):
            if 'Image' in project_name and 'Text' in project_name:                
                samples = pipe(
                    prompt_embeds=prompt_embed_text,
                    pooled_prompt_embeds=pooled_embed,
                    ip_adapter_image_embeds=prompt_embed_image,
                    guidance_scale=15,
                    negative_prompt="low quality, blurry, deformed, bad anatomy, disfigured, poorly drawn, mutated, extra limbs",
                    height=512,
                    width=512,
                    enable_router=True if lora_ckpt else False,
                    diffuse_router_image=diffuse_router_image,
                    diffuse_router_text=diffuse_router_text,
                    sos_embed=sos_embed,
                    text_embed_L=prompt_embed_text_L,
                    diffuse_prior_image=partial(prior_image, diffusion_prior=diffuse_prior_image),
                    diffuse_prior_text=partial(prior_text, diffusion_prior=diffuse_prior_text),
                    # denoising_end=0.8,
                    # output_type="latent",
                ).images
            elif 'Image' in project_name:
                samples = pipe(
                    prompt='',
                    ip_adapter_image_embeds=prompt_embed_image,
                    guidance_scale=15,
                    height=512,
                    width=512,
                    negative_prompt="low quality, blurry, deformed, bad anatomy, disfigured, poorly drawn, mutated, extra limbs",
                    enable_router=True if lora_ckpt else False,
                    diffuse_router_image=diffuse_router_image,
                    diffuse_prior_image=partial(prior_image, diffusion_prior=diffuse_prior_image),
                ).images
            else:
                samples = pipe(
                    prompt_embeds=prompt_embed_text,
                    pooled_prompt_embeds=pooled_embed,
                    guidance_scale=15,
                    height=512,
                    width=512,
                    negative_prompt="low quality, blurry, deformed, bad anatomy, disfigured, poorly drawn, mutated, extra limbs",
                    enable_router=True if lora_ckpt else False,
                    diffuse_router_text=diffuse_router_text,
                    sos_embed=sos_embed,
                    text_L_embed=prompt_embed_text_L,
                    diffuse_prior_text=partial(prior_text, diffusion_prior=diffuse_prior_text),
                ).images
            samples = torch.stack([transforms.PILToTensor()(s) for s in samples], 0)
            # samples = torch.stack([s for s in samples], 0)
            all_recons_val.append(samples)
            all_clip_img_val.append(np.array(clip_imgs))
            all_clip_txt_val.append(np.array(clip_txts))

        if (batch_idx + 1) % checkpoint_freq == 0 or batch_idx == total_val_batches - 1:
            val_progress["current_batch"] = batch_idx + 1
            val_progress["completed"] = (batch_idx == total_val_batches - 1)
            save_checkpoint(subj, project_name, model_name, device, "val", 
                        val_progress, image_indices_val, all_recons_val,
                        save_full=(batch_idx == total_val_batches - 1), clip_img=all_clip_img_val, clip_txt=all_clip_txt_val)
        accelerator.wait_for_everyone()
else:
    print("Validation phase already completed. Skipping.")

raise Exception

image_iters, voxel_iters, image_idx_iters = load_train_data(train_dls, voxels, subj_list, accelerator, batch_size, num_devices, num_sessions)
if not train_progress["completed"]:  
    # If resuming, try to load existing partial results
    if train_progress["current_step"] > 0:
        try:
            temp_indices_file = f"evals/{subj}/{project_name}/{model_name}/image_indices_train_temp_{device}.npy"
            recons_file = f"evals/{subj}/{project_name}/{model_name}/all_recons_train_{device}.pt"
            
            if os.path.exists(temp_indices_file) and os.path.exists(recons_file):
                image_indices_train = [np.load(temp_indices_file)]
                all_recons_train = [torch.load(recons_file)]
                print(f"Resuming training from step {train_progress['current_step']}")

            if 'Image' in project_name:
                temp_clip_img_file = f"evals/{subj}/{project_name}/{model_name}/clip_img_train_temp_{device}.npy"
                if os.path.exists(temp_clip_img_file):
                    all_clip_img_train = [np.load(temp_clip_img_file)]
            if 'Text' in project_name:
                temp_clip_txt_file = f"evals/{subj}/{project_name}/{model_name}/clip_txt_train_temp_{device}.npy"
                if os.path.exists(temp_clip_txt_file):
                    all_clip_txt_train = [np.load(temp_clip_txt_file)]
        except Exception as e:
            print(f"Error loading previous training results: {e}")
            train_progress["current_step"] = 0
            all_recons_train, image_indices_train = [], []

    with torch.no_grad():
        for step in tqdm(range(train_progress["current_step"], num_iterations_per_epoch), disable=not accelerator.is_main_process, desc="Train"):
            pixel_values, voxel_values, image_idx = [image_iters[s][step] for s in subj_list], [voxel_iters[s][step] for s in subj_list], [image_idx_iters[s][step].flatten() for s in subj_list]
            # pixel_values = torch.cat(pixel_values, dim=0)
            # pixel_values = pixel_values.to(accelerator.device)
            voxel_values = [voxel.to(accelerator.device) for voxel in voxel_values]
            image_indices_train.append(image_idx)
            
            with accelerator.autocast():
                if 'Image' in project_name:
                    prompt_embed_image, clip_img = get_image_embed(brain_model_image, diffuse_prior_image, voxel_values, subj_list, accelerator)
                if 'Text' in project_name:
                    prompt_embed_text, pooled_embed, clip_txt = get_text_embed(brain_model_text, diffuse_prior_text, voxel_values, subj_list, accelerator)
                
                    if 'TextL' in project_name:
                        prompt_embed_text_L, _, _ = get_text_embed(brain_model_text_L, diffuse_prior_text_L, voxel_values, subj_list, accelerator, text_L=True)

                    if not lora_ckpt:
                        if 'TextL' in project_name:
                            prompt_embed_text = torch.cat((prompt_embed_text_L, prompt_embed_text), dim=-1)
                        else:
                            prompt_embed_text = torch.cat((torch.zeros(prompt_embed_text.shape[0], 77, 768).to(accelerator.device), prompt_embed_text), dim=-1)
            
            with torch.amp.autocast(dtype=torch.float32, device_type='cuda'):
                if 'Image' in project_name and 'Text' in project_name:
                    if not lora_ckpt:  # we will do this inside SDXL pipeline'
                        prompt_embed_image = torch.cat((torch.zeros(prompt_embed_image.shape[0], 1, 1280).to(accelerator.device), prompt_embed_image), dim=0)

                    samples = pipe(
                        prompt_embeds=prompt_embed_text,
                        pooled_prompt_embeds=pooled_embed,
                        ip_adapter_image_embeds=prompt_embed_image,
                        guidance_scale=7.5,
                        negative_prompt="low quality, blurry, deformed, bad anatomy, disfigured, poorly drawn, mutated, extra limbs",
                        height=224,
                        width=224,
                        enable_router=True if lora_ckpt else False,
                        diffuse_router_image=diffuse_router_image,
                        diffuse_router_text=diffuse_router_text,
                        sos_embed=sos_embed,
                        text_L_embed=prompt_embed_text_L,
                        diffuse_prior_image=diffuse_prior_image,
                        diffuse_prior_text=diffuse_prior_text,
                    ).images
                elif 'Image' in project_name:
                    samples = []
                    for i in range(len(prompt_embed_image)):
                        samples.append(pipe(
                            prompt='',
                            ip_adapter_image_embeds=[torch.cat((torch.zeros(1, 1, 1280).to(accelerator.device), prompt_embed_image[i:i+1]), dim=0)],
                            guidance_scale=7.5,
                            height=224,
                            width=224,
                            negative_prompt="low quality, blurry, deformed, bad anatomy, disfigured, poorly drawn, mutated, extra limbs",
                            enable_router=True if lora_ckpt else False,
                            diffuse_router_image=diffuse_router_image,
                            diffuse_router_text=diffuse_router_text,
                            sos_embed=sos_embed,
                            text_L_embed=prompt_embed_text_L,
                            diffuse_prior_image=diffuse_prior_image,
                            diffuse_prior_text=diffuse_prior_text,
                        ).images[0])
                else:
                    samples = pipe(
                        prompt_embeds=prompt_embed_text,
                        pooled_prompt_embeds=pooled_embed,
                        guidance_scale=7.5,
                        height=224,
                        width=224,
                        negative_prompt="low quality, blurry, deformed, bad anatomy, disfigured, poorly drawn, mutated, extra limbs",
                        enable_router=True if lora_ckpt else False,
                        diffuse_router_image=diffuse_router_image,
                        diffuse_router_text=diffuse_router_text,
                        sos_embed=sos_embed,
                        text_L_embed=prompt_embed_text_L,
                        diffuse_prior_image=diffuse_prior_image,
                        diffuse_prior_text=diffuse_prior_text,
                    ).images
                samples = torch.stack([transforms.PILToTensor()(s) for s in samples], 0)
                all_recons_train.append(samples)
                if 'Image' in project_name: all_clip_img_train.append(np.array(clip_img.cpu()))
                if 'Text' in project_name: all_clip_txt_train.append(np.array(clip_txt.cpu()))
            
            if (step + 1) % checkpoint_freq == 0 or step == num_iterations_per_epoch - 1:
                train_progress["current_step"] = step + 1
                train_progress["completed"] = (step == num_iterations_per_epoch - 1)
                save_checkpoint(subj, project_name, model_name, device, "train", 
                                train_progress, image_indices_train, all_recons_train,
                                save_full=(step == num_iterations_per_epoch - 1), clip_img=all_clip_img_train, clip_txt=all_clip_txt_train)
            accelerator.wait_for_everyone()
else:
    print("Training phase already completed. Skipping.")