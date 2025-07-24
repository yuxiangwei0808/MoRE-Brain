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
import gc
import torch
import torch.nn as nn
from torchvision import transforms
from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler, DiffusionPipeline

# model_id = "stabilityai/stable-diffusion-2"
# scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
# pipe = StableDiffusionPipeline.from_pretrained(model_id, scheduler=scheduler, torch_dtype=torch.float16)

pipe = DiffusionPipeline.from_pretrained("ostris/Flex.1-alpha", torch_dtype=torch.float16, cache_dir='/home/users/ywei13/playground/BrainGen/data/cache')
# pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-3.5-large", torch_dtype=torch.float16, cache_dir='/home/users/ywei13/playground/BrainGen/data/cache')

pipe = pipe.to("cuda")
# pipe.enable_model_cpu_offload()
# pipe.enable_attention_slicing()

model_name = 'final_subj01_pretrained_40sess_24bsl_50_p_1.5'

all_predcaptions = torch.load(f"/home/users/ywei13/playground/BrainGen/checkpoints/MindEye2/evals/{model_name}/{model_name}_all_predcaptions.pt")
all_recons = torch.load(f"/home/users/ywei13/playground/BrainGen/checkpoints/MindEye2/evals/{model_name}/{model_name}_all_recons.pt") # these are the unrefined MindEye2 recons!

all_enhance_reconcs = []
for img_idx in tqdm(range(len(all_recons))):
    with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.float16):
        prompt = all_predcaptions[[img_idx]][0]
        image = pipe(prompt).images[0]
        all_enhance_reconcs.append(transforms.PILToTensor()(image).cpu())

        torch.cuda.empty_cache()
        gc.collect()

all_enhance_reconcs = torch.stack(all_enhance_reconcs)
torch.save(all_enhance_reconcs, f"/home/users/ywei13/playground/BrainGen/checkpoints/MindEye2/evals/{model_name}/flex1Alpha_all_enhancedrecons.pt")