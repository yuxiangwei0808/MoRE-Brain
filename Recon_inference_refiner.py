import os
import sys
import json
import argparse
import numpy as np
from einops import rearrange

from tqdm import tqdm
import argparse

import matplotlib.pyplot as plt
import torch
from torchvision import transforms
from accelerate import Accelerator
from diffusers import AutoPipelineForText2Image, AutoPipelineForImage2Image

sys.path.append('MindEyeV2/src/generative_models/')
sys.path.append('MindEyeV2/src')
from MindEyeV2.src.generative_models.sgm.models.diffusion import DiffusionEngine
from MindEyeV2.src.generative_models.sgm.util import append_dims

from brain_encoder import BrainEncoder, BrainDiffusionPriorEncoder
from brain_moe_encoder import BrainMoE, BrainMoEMulti, BrainMoEHier
from load_data import load_nsd
from routers import *

accelerator = Accelerator(split_batches=False, mixed_precision="fp16")
device = accelerator.device
world_size = accelerator.state.num_processes
distributed = not accelerator.state.distributed_type == 'NO'
num_devices = accelerator.state.num_processes
batch_size = 24
print("device:",device)


pipe_tmp = AutoPipelineForText2Image.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True, cache_dir='data/cache')
tokenizer_2, text_enc_2 = pipe_tmp.tokenizer_2, pipe_tmp.text_encoder_2
text_ids = tokenizer_2('a cat on a shelf', return_tensors="pt", padding="max_length", truncation=True, max_length=tokenizer_2.model_max_length)
sos_embed = text_enc_2(text_ids.input_ids, output_hidden_states=True).hidden_states[-2][:, :1]
tokenizer_1, text_enc_1 = pipe_tmp.tokenizer, pipe_tmp.text_encoder
text_ids = tokenizer_2('a cat on a shelf', return_tensors="pt", padding="max_length", truncation=True, max_length=tokenizer_1.model_max_length)
sos_embed_L = text_enc_1(text_ids.input_ids, output_hidden_states=True).hidden_states[-1][:, :1]


@torch.no_grad()
def get_text_embed(model, diffusion_prior, voxel_list, subj_list, accelerator, text_L=False):
    if text_L:
        backbones, clip_voxels, _, _ = model(voxel_list, subj_list, training=False, return_exp_out=False)    
    backbones, clip_voxels, _, _ = model(voxel_list, subj_list, training=False)
    
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


with open(f'data/NSD/coco_captions_phrases_5.json', 'r') as f:
    descs = json.load(f)

# refiner = DiffusionPipeline.from_pretrained(
refiner = AutoPipelineForImage2Image.from_pretrained(
    "stabilityai/stable-diffusion-xl-refiner-1.0",
    # text_encoder_2=base.text_encoder_2,
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16",
    cache_dir='data/cache',
)

refiner.to(device)
refiner._progress_bar_config = {'disable': True}

parser = argparse.ArgumentParser()
parser.add_argument('--noise_frac', type=float)
parser.add_argument('--strength', type=float, default=0.5)
args = parser.parse_args()

high_noise_frac = args.noise_frac
base_dir = f'evals/1/Text+Image-AblateParams/OutSize_512-minmaxNorm'
saving_dir = f"evals/1/Text+Image-refine/origin-prompt_phrase-strength{args.strength}-guide5"
os.makedirs(saving_dir, exist_ok=True)

base_recon = torch.load(os.path.join(base_dir, 'all_recons_val.pt'))
base_recon_idx = np.load(os.path.join(base_dir, 'all_indices_val.npy'))
assert len(base_recon) == len(base_recon_idx)

base_recon_dict = {}
for i in range(len(base_recon)):
    base_recon_dict[base_recon_idx[i]] = base_recon[i]

hidden_dim, clip_emb_dim, clip_seq_dim = 4096, 1280, 1
brain_model_text = BrainMoEMulti([15724], 4096, False, 4, clip_emb_dim, clip_seq_dim, 1, interm_out=False, enc_version='v1', num_exp_0=2, capacity_factor_0=1, num_exp_layer=3, exp_factor_list=[2, 2, 2], cap_fac_list=[1, 1, 1], meta=False)
diffuse_prior_text = BrainDiffusionPriorEncoder(clip_emb_dim, depth=6, dim_head=52, heads=clip_emb_dim//52, clip_seq_dim=clip_seq_dim, timesteps=100)

ckpt_text = torch.load(f'checkpoints/single_subj1_40sess_ViT_-cap5-prior-noblur-4096-BigG-BrainMoEMulti2-2L3F2--all+/best_mse.pth', map_location='cpu')
brain_model_text.load_state_dict(ckpt_text['model_state_dict'])
diffuse_prior_text.load_state_dict(ckpt_text['prior_state_dict'])
brain_model_text.requires_grad_(False)
diffuse_prior_text.requires_grad_(False)
brain_model_text.to(device).eval()
diffuse_prior_text.to(device).eval()
del ckpt_text

_, voxels, batch_size, test_dataloader, num_voxels_list, subj_list, num_iterations_per_epoch = load_nsd(40, 1, num_devices, batch_size, batch_size, False)
total_val_batches = len(test_dataloader)
test_dataloader = accelerator.prepare(test_dataloader)

all_recons = []
image_indices = []
for batch_idx, batch in enumerate(tqdm(test_dataloader, desc="Validation", disable=not accelerator.is_main_process)):
    pixel_values, voxel_values = batch['image'], batch['voxel'].to(accelerator.device)
    image_idx = batch['img_idx'].cpu().numpy()
    
    base_images = [base_recon_dict[image_idx[i]] for i in range(len(image_idx))]
    image_indices.append(image_idx)    
    
    prompts_txt = []
    prompts_txt_L = []
    with accelerator.autocast():
        for rep in range(3):
            prompt_embed_text, clip_txt, pooled_embed = get_text_embed(brain_model_text, diffuse_prior_text, [voxel_values[:, rep]], subj_list[:1], accelerator)            
            prompts_txt.append(prompt_embed_text)

        if isinstance(prompts_txt[0], list):
            prompt_embed_text = [torch.zeros_like(prompts_txt[0][i]) for i in range(len(prompts_txt[0]))]
            for elem in prompts_txt:
                for i in range(len(elem)):
                    prompt_embed_text[i] += elem[i]
            prompt_embed_text = [p / len(prompts_txt) for p in prompt_embed_text]
        else:
            prompt_embed_text = torch.stack(prompts_txt, 0).mean(0)

        prompt_embed_text = torch.cat([sos_embed.repeat(prompt_embed_text.shape[0], 1, 1).to(accelerator.device), prompt_embed_text.repeat(1, 76, 1)], dim=1)
        pooled_embed = torch.zeros_like(prompt_embed_text).mean(1).to(accelerator.device)
        
        with torch.amp.autocast(dtype=torch.float32, device_type='cuda'):
            recons = []
            for i in range(len(base_images)):
                image = refiner(
                    prompt_embeds=prompt_embed_text[i:i+1],
                    pooled_prompt_embeds=pooled_embed[i:i+1],
                    negative_prompt="painting, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, deformed, ugly, blurry, bad anatomy, bad proportions, extra limbs, cloned face, skinny, glitchy, double torso, extra arms, extra hands, mangled fingers, missing lips, ugly face, distorted face, extra legs, anime",
                    denoising_start=high_noise_frac,
                    image=base_images[i],
                    strength=args.strength,
                    guidance_scale=5,
                ).images[0] 
                recons.append(image)
            recons = torch.stack([transforms.PILToTensor()(s) for s in recons], 0)
            all_recons.append(recons)

    torch.save(all_recons, f"{saving_dir}/all_recons_val_{device}.pt")
    np.save(f"{saving_dir}/image_indices_val_{device}.npy", np.concatenate(image_indices))
    accelerator.wait_for_everyone()
