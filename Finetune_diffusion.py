#!/usr/bin/env python
# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
"""Fine-tuning script for Stable Diffusion for text2image with support for LoRA."""

import argparse
import logging
import math
import os
import random
import shutil
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, List, Union, Optional

import datasets
import numpy as np
import torch
import torch.nn.functional as F
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs, DistributedType, ProjectConfiguration, set_seed
from datasets import load_dataset
from huggingface_hub import create_repo, upload_folder
from packaging import version
from peft import LoraConfig, set_peft_model_state_dict
from peft.utils import get_peft_model_state_dict
from torchvision import transforms
from torchvision.transforms.functional import crop
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PretrainedConfig, CLIPImageProcessor
from einops import rearrange

import diffusers
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    StableDiffusionXLPipeline,
    UNet2DConditionModel,
)
from diffusers.loaders import StableDiffusionLoraLoaderMixin
from diffusers.optimization import get_scheduler
from diffusers.training_utils import _set_state_dict_into_text_encoder, cast_training_params, compute_snr
from diffusers.utils import (
    check_min_version,
    convert_state_dict_to_diffusers,
    convert_unet_state_dict_to_peft,
    is_wandb_available,
    _get_model_file,
)
from diffusers.models.modeling_utils import _LOW_CPU_MEM_USAGE_DEFAULT, load_state_dict
from diffusers.utils.hub_utils import load_or_create_model_card, populate_model_card
from diffusers.utils.import_utils import is_torch_npu_available, is_xformers_available
from diffusers.utils.torch_utils import is_compiled_module

import wandb
import sys
sys.path.append('MindEyeV2/src/generative_models/')
sys.path.append('MindEyeV2/src')
from load_data import load_nsd, load_train_data
import MindEyeV2.src.utils as utils
from brain_encoder import BrainEncoder, BrainDiffusionPriorEncoder
from brain_moe_encoder import BrainMoE, BrainMoEMulti
from routers import *

logger = get_logger(__name__, log_level="INFO")

# Global tokenizers and embeddings for text processing
image_processor = CLIPImageProcessor.from_pretrained(
    "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k", 
    cache_dir="data/cache"
)

# Initialize pipeline for tokenizers
pipeline = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", 
    torch_dtype=torch.float16, 
    variant="fp16", 
    use_safetensors=True, 
    cache_dir='data/cache'
)

# Extract tokenizers and text encoders
tokenizer_2, text_enc_2 = pipeline.tokenizer_2, pipeline.text_encoder_2
tokenizer_1, text_enc_1 = pipeline.tokenizer, pipeline.text_encoder

# Create embeddings for text processing
text_ids_2 = tokenizer_2(
    'a cat on a shelf', 
    return_tensors="pt", 
    padding="max_length", 
    truncation=True, 
    max_length=tokenizer_2.model_max_length
)
sos_embed = text_enc_2(text_ids_2.input_ids, output_hidden_states=True).hidden_states[-2][:, :1]

text_ids_1 = tokenizer_1(
    'a cat on a shelf', 
    return_tensors="pt", 
    padding="max_length", 
    truncation=True, 
    max_length=tokenizer_1.model_max_length
)
sos_embed_L = text_enc_1(text_ids_1.input_ids, output_hidden_states=True).hidden_states[-1][:, :1]


def process_image(image):
    """Process image tensor and convert to PIL format for CLIP processing."""
    pil_images = []
    for i in range(image.shape[0]):
        img = image[i].permute(1, 2, 0).cpu().numpy()
        pil_img = transforms.ToPILImage()(img)
        pil_images.append(pil_img)
    
    return image_processor(pil_images, return_tensors="pt").pixel_values


def save_model_card(
    repo_id: str,
    images: list = None,
    base_model: str = None,
    dataset_name: str = None,
    train_text_encoder: bool = False,
    repo_folder: str = None,
    vae_path: str = None,
):
    img_str = ""
    if images is not None:
        for i, image in enumerate(images):
            image.save(os.path.join(repo_folder, f"image_{i}.png"))
            img_str += f"![img_{i}](./image_{i}.png)\n"

    model_description = f"""
# LoRA text2image fine-tuning - {repo_id}

These are LoRA adaption weights for {base_model}. The weights were fine-tuned on the {dataset_name} dataset. You can find some example images in the following. \n
{img_str}

LoRA for the text encoder was enabled: {train_text_encoder}.

Special VAE used for training: {vae_path}.
"""
    model_card = load_or_create_model_card(
        repo_id_or_path=repo_id,
        from_training=True,
        license="creativeml-openrail-m",
        base_model=base_model,
        model_description=model_description,
        inference=True,
    )

    tags = [
        "stable-diffusion-xl",
        "stable-diffusion-xl-diffusers",
        "text-to-image",
        "diffusers",
        "diffusers-training",
        "lora",
    ]
    model_card = populate_model_card(model_card, tags=tags)

    model_card.save(os.path.join(repo_folder, "README.md"))


def log_validation(
    pipeline,
    args,
    accelerator,
    epoch,
    is_final_validation=False,
):
    logger.info(
        f"Running validation... \n Generating {args.num_validation_images} images with prompt:"
        f" {args.validation_prompt}."
    )
    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)

    # run inference
    generator = torch.Generator(device=accelerator.device).manual_seed(args.seed) if args.seed is not None else None
    pipeline_args = {"prompt": args.validation_prompt}
    if torch.backends.mps.is_available():
        autocast_ctx = nullcontext()
    else:
        autocast_ctx = torch.autocast(accelerator.device.type)

    with autocast_ctx:
        images = [pipeline(**pipeline_args, generator=generator).images[0] for _ in range(args.num_validation_images)]

    for tracker in accelerator.trackers:
        phase_name = "test" if is_final_validation else "validation"
        if tracker.name == "tensorboard":
            np_images = np.stack([np.asarray(img) for img in images])
            tracker.writer.add_images(phase_name, np_images, epoch, dataformats="NHWC")
        if tracker.name == "wandb":
            tracker.log(
                {
                    phase_name: [
                        wandb.Image(image, caption=f"{i}: {args.validation_prompt}") for i, image in enumerate(images)
                    ]
                }
            )
    return images


def import_model_class_from_model_name_or_path(
    pretrained_model_name_or_path: str, revision: str, subfolder: str = "text_encoder"
):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path, subfolder=subfolder, revision=revision
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "CLIPTextModelWithProjection":
        from transformers import CLIPTextModelWithProjection

        return CLIPTextModelWithProjection
    else:
        raise ValueError(f"{model_class} is not supported.")

def load_ip_adapter(
        unet,
        pretrained_model_name_or_path_or_dict: Union[str, List[str], Dict[str, torch.Tensor]],
        subfolder: Union[str, List[str]],
        weight_name: Union[str, List[str]],
        image_encoder_folder: Optional[str] = "image_encoder",
        **kwargs,
    ):
    # adapted from diffusers.loaders.ip_adapter.IPAdapterMixin
    # handle the list inputs for multiple IP Adapters
    if not isinstance(weight_name, list):
        weight_name = [weight_name]

    if not isinstance(pretrained_model_name_or_path_or_dict, list):
        pretrained_model_name_or_path_or_dict = [pretrained_model_name_or_path_or_dict]
    if len(pretrained_model_name_or_path_or_dict) == 1:
        pretrained_model_name_or_path_or_dict = pretrained_model_name_or_path_or_dict * len(weight_name)

    if not isinstance(subfolder, list):
        subfolder = [subfolder]
    if len(subfolder) == 1:
        subfolder = subfolder * len(weight_name)

    if len(weight_name) != len(pretrained_model_name_or_path_or_dict):
        raise ValueError("`weight_name` and `pretrained_model_name_or_path_or_dict` must have the same length.")

    if len(weight_name) != len(subfolder):
        raise ValueError("`weight_name` and `subfolder` must have the same length.")

    # Load the main state dict first.
    cache_dir = kwargs.pop("cache_dir", None)
    force_download = kwargs.pop("force_download", False)
    proxies = kwargs.pop("proxies", None)
    local_files_only = kwargs.pop("local_files_only", None)
    token = kwargs.pop("token", None)
    revision = kwargs.pop("revision", None)
    low_cpu_mem_usage = False

    if low_cpu_mem_usage and not is_accelerate_available():
        low_cpu_mem_usage = False
        logger.warning(
            "Cannot initialize model with low cpu memory usage because `accelerate` was not found in the"
            " environment. Defaulting to `low_cpu_mem_usage=False`. It is strongly recommended to install"
            " `accelerate` for faster and less memory-intense model loading. You can do so with: \n```\npip"
            " install accelerate\n```\n."
        )

    if low_cpu_mem_usage is True and not is_torch_version(">=", "1.9.0"):
        raise NotImplementedError(
            "Low memory initialization requires torch >= 1.9.0. Please either update your PyTorch version or set"
            " `low_cpu_mem_usage=False`."
        )

    user_agent = {
        "file_type": "attn_procs_weights",
        "framework": "pytorch",
    }
    state_dicts = []
    for pretrained_model_name_or_path_or_dict, weight_name, subfolder in zip(
        pretrained_model_name_or_path_or_dict, weight_name, subfolder
    ):
        if not isinstance(pretrained_model_name_or_path_or_dict, dict):
            model_file = _get_model_file(
                pretrained_model_name_or_path_or_dict,
                weights_name=weight_name,
                cache_dir=cache_dir,
                force_download=force_download,
                proxies=proxies,
                local_files_only=local_files_only,
                token=token,
                revision=revision,
                subfolder=subfolder,
                user_agent=user_agent,
            )
            if weight_name.endswith(".safetensors"):
                state_dict = {"image_proj": {}, "ip_adapter": {}}
                with safe_open(model_file, framework="pt", device="cpu") as f:
                    for key in f.keys():
                        if key.startswith("image_proj."):
                            state_dict["image_proj"][key.replace("image_proj.", "")] = f.get_tensor(key)
                        elif key.startswith("ip_adapter."):
                            state_dict["ip_adapter"][key.replace("ip_adapter.", "")] = f.get_tensor(key)
            else:
                state_dict = load_state_dict(model_file)
        else:
            state_dict = pretrained_model_name_or_path_or_dict

        keys = list(state_dict.keys())
        if "image_proj" not in keys and "ip_adapter" not in keys:
            raise ValueError("Required keys are (`image_proj` and `ip_adapter`) missing from the state dict.")

        state_dicts.append(state_dict)

    # load ip-adapter into unet
    unet._load_ip_adapter_weights(state_dicts, low_cpu_mem_usage=low_cpu_mem_usage)


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default='stabilityai/stable-diffusion-xl-base-1.0',
        # required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--pretrained_vae_model_name_or_path",
        type=str,
        default=None,
        help="Path to pretrained VAE model with better numerical stability. More details: https://github.com/huggingface/diffusers/pull/4038.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default='fp16',
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The config of the Dataset, leave as None if there's only one config.",
    )
    parser.add_argument(
        "--validation_prompt", type=str, default=None, help="A prompt that is sampled during training for inference."
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=4,
        help="Number of images that should be generated during validation with `validation_prompt`.",
    )
    parser.add_argument(
        "--validation_epochs",
        type=int,
        default=1,
        help=(
            "Run fine-tuning validation every X epochs. The validation process consists of running the prompt"
            " `args.validation_prompt` multiple times: `args.num_validation_images`."
        ),
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="checkpoints/sd-model-finetuned-lora",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default='data/cache',
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--center_crop",
        default=False,
        action="store_true",
        help=(
            "Whether to center crop the input images to the resolution. If not set, the images will be randomly"
            " cropped. The images will be resized to the resolution first before cropping."
        ),
    )
    parser.add_argument(
        "--random_flip",
        action="store_true",
        help="whether to randomly flip images horizontally",
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=48, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument(
        "--test_batch_size", type=int, default=48, help="Batch size (per device) for the test dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant_with_warmup",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--snr_gamma",
        type=float,
        default=None,
        help="SNR weighting gamma to be used if rebalancing the loss. Recommended value is 5.0. "
        "More details here: https://arxiv.org/abs/2303.09556.",
    )
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=16,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--prediction_type",
        type=str,
        default=None,
        help="The prediction_type that shall be used for training. Choose between 'epsilon' or 'v_prediction' or leave `None`. If left to `None` the default prediction type of the scheduler: `noise_scheduler.config.prediction_type` is chosen.",
    )
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default='fp16',
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="wandb",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )
    parser.add_argument("--noise_offset", type=float, default=0, help="The scale of noise offset.")
    parser.add_argument(
        "--rank",
        type=int,
        default=4,
        help=("The dimension of the LoRA update matrices."),
    )
    parser.add_argument('--train_text_encoder', action='store_true', help='Whether to train the text encoder')
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
        "--subj",type=int, default=1, choices=[1,2,3,4,5,6,7,8],
        help="Validate on which subject?",
    )
    parser.add_argument(
        "--num_sessions", type=int, default=40,
        help="Number of training sessions to include",
    )
    parser.add_argument(
        "--wandb_log",action=argparse.BooleanOptionalAction, default=False,
        help="whether to log to wandb",
    )
    parser.add_argument(
        "--multi_subject",action=argparse.BooleanOptionalAction,default=False,
    )
    parser.add_argument(
        '--route_image', action='store_true', default=True, help='use routers for the fMRI-CLIP Image decoder; obtain outputs from each level of experts then apply the router'
    )
    parser.add_argument(
        '--route_text', action='store_true', default=True, help='use routers for the fMRI-CLIP Text decoder; obtain outputs from each level of experts then apply the router'
    )
    parser.add_argument(
        '--logging_name', type=str, default='',
    )
    parser.add_argument(
        '--tune_trained_routers', action='store_true', default=False, help='tune routers that are train in the MoE model'
    )
    parser.add_argument(
        '--b_size', type=int, default=-1, help='bottleneck size'
    )
    parser.add_argument(
        '--resized_pixel_size', type=int, default=224,
    )

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args


DATASET_NAME_MAPPING = {
    "lambdalabs/naruto-blip-captions": ("image", "text"),
}

def init_wandb(args):
    wandb_project = 'brainGen-tune-sdxl'
    # need to configure wandb beforehand in terminal with "wandb init"!
    wandb_config = {
        "model_name_image": args.model_name_image,
        "model_name_text": args.model_name_text,    
        "batch_size": args.train_batch_size,
        "num_sessions": args.num_sessions,
        "max_lr": args.learning_rate,
        "seed": args.seed,
        "subj": args.subj,
        "multi_subject": args.multi_subject,
    }
    wandb_config.update({
        "unet_lora_rank": args.rank,
        "snr_gamma": args.snr_gamma,
        "noise_offset": args.noise_offset,
        "route_image": args.route_image,
        "route_text": args.route_text,
        "resolution": args.resolution,
    })
    print("wandb_config:\n", wandb_config)
    args.logging_name = args.output_dir.split("/")[-2]
    # args.logging_name = f"{args.model_name_image}---{args.model_name_text}"
    wandb.init(
        # id=model_name,
        project=wandb_project,
        name=args.logging_name,
        config=wandb_config,
        # resume="allow",
    )
    wandb.run.log_code(".", include_fn=lambda path: path.endswith(".py"))


@torch.no_grad()
def get_image_embed(model, diffusion_prior, voxel_list, subj_list, accelerator, args):
    if 'MoE' in args.model_name_image:
        backbones, _, _, _ = model(voxel_list, subj_list, training=False, return_exp_out=True if args.route_image else False)
    else:
        backbones, _, _ = model(voxel_list, subj_list)
    return backbones

@torch.no_grad()
def prior_image(backbones, diffusion_prior, accelerator):
    # prior_out = diffusion_prior.prior.p_sample_loop(backbones.shape, text_cond=dict(text_embed=backbones), cond_scale=1.5, timesteps=30)
    _, prior_out = diffusion_prior(text_embed=backbones, image_embed=torch.zeros_like(backbones).to(backbones.device))
    return [prior_out]

@torch.no_grad()
def get_text_embed(model, diffusion_prior, voxel_list, subj_list, accelerator, args):
    if 'MoE' in args.model_name_text:
        backbones, _, _, _ = model(voxel_list, subj_list, training=False, return_exp_out=True if args.route_text else False)
    else:
        backbones, _, _ = model(voxel_list, subj_list)
    
    return backbones

@torch.no_grad()
def prior_text(backbones, diffusion_prior, accelerator):
    bz, seq_len = backbones.shape[0], backbones.shape[-1]
    # prior_out = diffusion_prior.prior.p_sample_loop(backbones.shape, text_cond=dict(text_embed=backbones), cond_scale=1.5, timesteps=30)
    _, prior_out = diffusion_prior(text_embed=backbones, image_embed=torch.zeros_like(backbones).to(backbones.device))
    pooled_embed = torch.zeros(bz, seq_len).to(accelerator.device)

    return prior_out, pooled_embed

def save_routers(accelerator, outdir, tag, diffuse_router_image, diffuse_router_text):
    if diffuse_router_image is not None:
        unwrapped_model = accelerator.unwrap_model(diffuse_router_image)
        torch.save(unwrapped_model.state_dict(), os.path.join(outdir, f"{tag}_diffuse_router_image.pth"))
    if diffuse_router_text is not None:
        unwrapped_model = accelerator.unwrap_model(diffuse_router_text)
        torch.save(unwrapped_model.state_dict(), os.path.join(outdir, f"{tag}_diffuse_router_text.pth"))
    print(f"Saved routers to {outdir}")

def save_prior(accelerator, outdir, tag, diffuse_prior_image, diffuse_prior_text):
    if diffuse_prior_image.requires_grad_:
        unwrapped_model = accelerator.unwrap_model(diffuse_prior_image)
        torch.save(unwrapped_model.state_dict(), os.path.join(outdir, f"{tag}_diffuse_prior_image.pth"))
    if diffuse_prior_text.requires_grad_:
        unwrapped_model = accelerator.unwrap_model(diffuse_prior_text)
        torch.save(unwrapped_model.state_dict(), os.path.join(outdir, f"{tag}_diffuse_prior_text.pth"))
    print(f"Saved routers to {outdir}")

def transfer_router_weights(
    moe_model,
    diffuse_router,
):    
    for idx, back_router in enumerate(moe_model.back_router):
        # Check if the router types match
        if not isinstance(diffuse_router.space_routers[idx], type(back_router)):
            raise Exception(f"Warning: Router type mismatch at index {idx}. Source: {type(back_router)}, Target: {type(diffuse_router.space_routers[idx])}")
        
        diffuse_router.space_routers[idx].load_state_dict(back_router.state_dict())
    print(f"Successfully transferred weights from {len(moe_model.back_router)} routers")
    return diffuse_router


def main():
    args = parse_args()
    if args.report_to == "wandb" and args.hub_token is not None:
        raise ValueError(
            "You cannot use both --report_to=wandb and --hub_token due to a security risk of exposing your token."
            " Please use `huggingface-cli login` to authenticate with the Hub."
        )

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        split_batches=False,
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True),]
        # log_with=args.report_to,
        # project_config=accelerator_project_config,
    )
    num_devices = accelerator.num_processes

    # Disable AMP for MPS.
    if torch.backends.mps.is_available():
        accelerator.native_amp = False

    # Make one log on every process with the configuration for debugging.
    if args.wandb_log and accelerator.is_local_main_process:
        init_wandb(args)

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name, exist_ok=True, token=args.hub_token
            ).repo_id

    train_dls, voxels, args.train_batch_size, test_dataloader, num_voxels_list, subj_list, num_iterations_per_epoch = load_nsd(args.num_sessions, args.subj, num_devices, args.train_batch_size, args.test_batch_size, args.multi_subject)

    # Load scheduler, tokenizer and models.
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    # tokenizer = CLIPTokenizer.from_pretrained(
    #     args.pretrained_model_name_or_path, subfolder="tokenizer", revision=args.revision
    # )
    # text_encoder = CLIPTextModel.from_pretrained(
    #     args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision
    # )
    vae_path = (
        args.pretrained_model_name_or_path
        if args.pretrained_vae_model_name_or_path is None
        else args.pretrained_vae_model_name_or_path
    )
    vae = AutoencoderKL.from_pretrained(
        vae_path,
        subfolder="vae" if args.pretrained_vae_model_name_or_path is None else None,
        revision=args.revision,
        variant=args.variant,
        cache_dir=args.cache_dir,
    )
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision, variant=args.variant, cache_dir=args.cache_dir
    )
    load_ip_adapter(unet, "h94/IP-Adapter", subfolder="sdxl_models", weight_name="ip-adapter_sdxl.bin", cache_dir='data/cache',)
    
    hidden_dim, clip_emb_dim, clip_seq_dim = 4096, 1280, 1
    # brain_model_text = BrainEncoder(num_voxels_list, hidden_dim, False, 4, clip_emb_dim, clip_seq_dim, 1, interm_out=False, enc_version='v1')
    brain_model_text = BrainMoEMulti(num_voxels_list, hidden_dim, False, 4, clip_emb_dim, clip_seq_dim, 1, interm_out=False, enc_version='v1', num_exp_0=2, capacity_factor_0=1, num_exp_layer=3, exp_factor_list=[2, 2, 2], cap_fac_list=[1, 1, 1], meta=False, b_size=args.b_size)
    diffuse_prior_text = BrainDiffusionPriorEncoder(clip_emb_dim, depth=6, dim_head=52, heads=clip_emb_dim//52, clip_seq_dim=clip_seq_dim, timesteps=100)

    if args.model_name_text_L:
        hidden_dim, clip_emb_dim, clip_seq_dim = 4096, 768, 1
        brain_model_text_L = BrainEncoder(num_voxels_list, hidden_dim, False, 4, clip_emb_dim, clip_seq_dim, 1, interm_out=False, enc_version='v1')
        diffuse_prior_text_L = BrainDiffusionPriorEncoder(clip_emb_dim, depth=6, dim_head=52, heads=clip_emb_dim//52, clip_seq_dim=clip_seq_dim, timesteps=100)

    # brain_model_image = BrainEncoder(num_voxels_list, hidden_dim, False, 4, clip_emb_dim, clip_seq_dim, 1, interm_out=False, enc_version='v1')
    brain_model_image = BrainMoEMulti(num_voxels_list, hidden_dim, False, 4, clip_emb_dim, clip_seq_dim, 1, interm_out=False, enc_version='v1', num_exp_0=2, capacity_factor_0=1, num_exp_layer=3, exp_factor_list=[2, 2, 2], cap_fac_list=[1, 1, 1], meta=False, b_size=args.b_size)
    diffuse_prior_image = BrainDiffusionPriorEncoder(clip_emb_dim, depth=6, dim_head=52, heads=clip_emb_dim//52, clip_seq_dim=clip_seq_dim, timesteps=100)

    if args.route_image: 
        diffuse_router_image = DiffuseRouter(
            time_router_class=DiffuseTimeRouterAttn,
            space_router_class=DiffuseSpaceRouterCrossAttn,
            num_granularity_levels=4,
            num_experts_per_granularity=[2, 4, 8, 16],
        )
    if args.route_text: 
        diffuse_router_text = DiffuseRouter(
            time_router_class=DiffuseTimeRouterAttn,
            space_router_class=DiffuseSpaceRouterCrossAttn,
            num_granularity_levels=4,
            num_experts_per_granularity=[2, 4, 8, 16],
        )

    # Loading pretrained models
    ckpt_image = torch.load(f'checkpoints/{args.model_name_image}/best_mse.pth', map_location='cpu')
    ckpt_text = torch.load(f'checkpoints/{args.model_name_text}/best_mse.pth', map_location='cpu')
    brain_model_image.load_state_dict(ckpt_image['model_state_dict'])
    brain_model_text.load_state_dict(ckpt_text['model_state_dict'])
    diffuse_prior_image.load_state_dict(ckpt_image['prior_state_dict'])
    diffuse_prior_text.load_state_dict(ckpt_text['prior_state_dict'])

    if args.tune_trained_routers:
        assert not brain_model_image.routing and not brain_model_text.routing, "Routing should be disabled when tunining"
        transfer_router_weights(brain_model_image, diffuse_router_image)
        transfer_router_weights(brain_model_text, diffuse_router_text)

    del ckpt_image, ckpt_text

    if args.model_name_text_L:
        ckpt_text_L = torch.load(f'checkpoints/{args.model_name_text_L}/best_mse.pth', map_location='cpu')
        brain_model_text_L.load_state_dict(ckpt_text_L['model_state_dict'])
        diffuse_prior_text_L.load_state_dict(ckpt_text_L['prior_state_dict'])
        del ckpt_text_L

    # freeze parameters of models to save more memory
    unet.requires_grad_(False)
    vae.requires_grad_(False)
    # text_encoder.requires_grad_(False)
    brain_model_text.requires_grad_(False)
    brain_model_image.requires_grad_(False)
    diffuse_prior_image.requires_grad_(False)
    diffuse_prior_text.requires_grad_(False)
    if args.model_name_text_L: 
        brain_model_text_L.requires_grad_(False)
        diffuse_prior_text_L.requires_grad_(False)

    # For mixed precision training we cast all non-trainable weights (vae, non-lora text_encoder and non-lora unet) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move unet, vae and text_encoder to device and cast to weight_dtype
    unet.to(accelerator.device, dtype=weight_dtype)
    if args.pretrained_vae_model_name_or_path is None:
        vae.to(accelerator.device, dtype=torch.float32)
    else:
        vae.to(accelerator.device, dtype=weight_dtype)
    # text_encoder.to(accelerator.device, dtype=weight_dtype)
    brain_model_image.to(accelerator.device, dtype=weight_dtype)
    brain_model_text.to(accelerator.device, dtype=weight_dtype)
    diffuse_prior_image.to(accelerator.device, dtype=weight_dtype)
    diffuse_prior_text.to(accelerator.device, dtype=weight_dtype)
    if args.model_name_text_L: 
        brain_model_text_L.to(accelerator.device, dtype=weight_dtype)
        diffuse_prior_text_L.to(accelerator.device, dtype=weight_dtype)
    
    if args.route_image: diffuse_router_image.to(accelerator.device, dtype=torch.float32)
    if args.route_text: diffuse_router_text.to(accelerator.device, dtype=torch.float32)

    # Add adapter and make sure the trainable params are in float32.
    unet_lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.rank,
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0", 'ff.net.0.proj', 'ff.net.2'],  # proj_in, proj_out, ff.net.0.proj, ff.net.2
        # use_rslora=True,
    )
    unet.add_adapter(unet_lora_config)
    if args.mixed_precision == "fp16":
        # only upcast trainable parameters (LoRA) into fp32
        cast_training_params(unet, dtype=torch.float32)

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warning(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    params_to_optimize = list(filter(lambda p: p.requires_grad, unet.parameters()))
    params_to_optimize = params_to_optimize
    if args.route_image: params_to_optimize += list(filter(lambda p: p.requires_grad, diffuse_router_image.parameters()))
    if args.route_text: params_to_optimize += list(filter(lambda p: p.requires_grad, diffuse_router_text.parameters()))
    if diffuse_prior_image.requires_grad_: params_to_optimize += list(filter(lambda p: p.requires_grad, diffuse_prior_image.parameters()))
    if diffuse_prior_text.requires_grad_: params_to_optimize += list(filter(lambda p: p.requires_grad, diffuse_prior_text.parameters()))

    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    def save_model_hook(models, weights, output_dir):
        if accelerator.is_main_process:
            # there are only two options here. Either are just the unet attn processor layers
            # or there are the unet and text encoder attn layers
            unet_lora_layers_to_save = None
            text_encoder_one_lora_layers_to_save = None
            text_encoder_two_lora_layers_to_save = None

            for model in models:
                if isinstance(unwrap_model(model), type(unwrap_model(unet))):
                    unet_lora_layers_to_save = convert_state_dict_to_diffusers(get_peft_model_state_dict(model))
                else:
                    raise ValueError(f"unexpected save model: {model.__class__}")

                # make sure to pop weight so that corresponding model is not saved again
                if weights:
                    weights.pop()

            StableDiffusionXLPipeline.save_lora_weights(
                output_dir,
                unet_lora_layers=unet_lora_layers_to_save,
                text_encoder_lora_layers=text_encoder_one_lora_layers_to_save,
                text_encoder_2_lora_layers=text_encoder_two_lora_layers_to_save,
            )

    def load_model_hook(models, input_dir):
        unet_ = None
        text_encoder_one_ = None
        text_encoder_two_ = None

        while len(models) > 0:
            model = models.pop()

            if isinstance(model, type(unwrap_model(unet))):
                unet_ = model
            elif isinstance(model, type(unwrap_model(text_encoder_one))):
                text_encoder_one_ = model
            elif isinstance(model, type(unwrap_model(text_encoder_two))):
                text_encoder_two_ = model
            else:
                raise ValueError(f"unexpected save model: {model.__class__}")

        lora_state_dict, _ = StableDiffusionLoraLoaderMixin.lora_state_dict(input_dir)
        unet_state_dict = {f'{k.replace("unet.", "")}': v for k, v in lora_state_dict.items() if k.startswith("unet.")}
        unet_state_dict = convert_unet_state_dict_to_peft(unet_state_dict)
        incompatible_keys = set_peft_model_state_dict(unet_, unet_state_dict, adapter_name="default")
        if incompatible_keys is not None:
            # check only for unexpected keys
            unexpected_keys = getattr(incompatible_keys, "unexpected_keys", None)
            if unexpected_keys:
                logger.warning(
                    f"Loading adapter weights from state_dict led to unexpected keys not found in the model: "
                    f" {unexpected_keys}. "
                )

        if args.train_text_encoder:
            _set_state_dict_into_text_encoder(lora_state_dict, prefix="text_encoder.", text_encoder=text_encoder_one_)

            _set_state_dict_into_text_encoder(
                lora_state_dict, prefix="text_encoder_2.", text_encoder=text_encoder_two_
            )

        # Make sure the trainable params are in float32. This is again needed since the base models
        # are in `weight_dtype`. More details:
        # https://github.com/huggingface/diffusers/pull/6514#discussion_r1449796804
        if args.mixed_precision == "fp16":
            models = [unet_]
            if args.train_text_encoder:
                models.extend([text_encoder_one_, text_encoder_two_])
            cast_training_params(models, dtype=torch.float32)

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Initialize the optimizer
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    optimizer = optimizer_cls(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # Get the datasets: you can either provide your own training and evaluation files (see below)
    # or specify a Dataset from the hub (the dataset will be downloaded automatically from the datasets Hub).

    if args.resized_pixel_size != 224:
        pixel_transforms = transforms.Resize((args.resized_pixel_size, args.resized_pixel_size))
    else:
        pixel_transforms = None

    # with accelerator.main_process_first():
    #     if args.max_train_samples is not None:
    #         dataset["train"] = dataset["train"].shuffle(seed=args.seed).select(range(args.max_train_samples))
    #     # Set the training transforms
    #     train_dataset = dataset["train"].with_transform(preprocess_train)

    # Scheduler and math around the number of training steps.
    # Check the PR https://github.com/huggingface/diffusers/pull/8312 for detailed explanation.
    num_warmup_steps_for_scheduler = args.lr_warmup_steps * accelerator.num_processes
    if args.max_train_steps is None:
        len_train_dataloader_after_sharding = math.ceil(num_iterations_per_epoch / accelerator.num_processes)
        num_update_steps_per_epoch = math.ceil(len_train_dataloader_after_sharding / args.gradient_accumulation_steps)
        num_training_steps_for_scheduler = (
            args.num_train_epochs * num_update_steps_per_epoch * accelerator.num_processes
        )
    else:
        num_training_steps_for_scheduler = args.max_train_steps * accelerator.num_processes

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps_for_scheduler,
        num_training_steps=num_training_steps_for_scheduler,
    )

    # Prepare everything with our `accelerator`.
    train_dls = train_dls.values()
    brain_model_text, brain_model_image, diffuse_prior_image, diffuse_prior_text, unet, optimizer, lr_scheduler, *train_dls = accelerator.prepare(
        brain_model_text, brain_model_image, diffuse_prior_image, diffuse_prior_text, unet, optimizer, lr_scheduler, *train_dls
    )
    if args.route_image: diffuse_router_image = accelerator.prepare(diffuse_router_image)
    if args.route_text: diffuse_router_text = accelerator.prepare(diffuse_router_text)
    if args.model_name_text_L: brain_model_text_L, diffuse_prior_text_L = accelerator.prepare(brain_model_text_L, diffuse_prior_text_L)
    
    image_iters, voxel_iters, image_idx_iters = load_train_data(train_dls, voxels, subj_list, accelerator, args.train_batch_size, num_devices, args.num_sessions, num_iterations_per_epoch)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(num_iterations_per_epoch / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        if num_training_steps_for_scheduler != args.max_train_steps * accelerator.num_processes:
            logger.warning(
                f"The length of the 'train_dataloader' after 'accelerator.prepare' ({num_iterations_per_epoch}) does not match "
                f"the expected length ({len_train_dataloader_after_sharding}) when the learning rate scheduler was created. "
                f"This inconsistency may result in the learning rate scheduler not functioning properly."
            )
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("text2image-fine-tune", config=vars(args))

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num Iters = {num_iterations_per_epoch}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch
    else:
        initial_global_step = 0

    best_val_loss = float('inf')
    for epoch in range(first_epoch, args.num_train_epochs):
        progress_bar = tqdm(
            range(0, num_iterations_per_epoch),
            initial=initial_global_step,
            desc="Steps",
            # Only show the progress bar once on each machine.
            disable=not accelerator.is_local_main_process,
        )
        unet.train()

        if args.route_image: diffuse_router_image.train()
        if args.route_text: diffuse_router_text.train()
        train_loss = 0.0
        for step in range(num_iterations_per_epoch):
            with accelerator.accumulate(unet):
                pixel_values, voxel_values, image_idx = [image_iters[s][step] for s in subj_list], [voxel_iters[s][step] for s in subj_list], [image_idx_iters[s][step] for s in subj_list]
                pixel_values = torch.cat(pixel_values, dim=0)
                pixel_values = pixel_values.to(accelerator.device)
                voxel_values = [voxel.to(accelerator.device) for voxel in voxel_values]

                if pixel_transforms is not None:
                    pixel_values = pixel_transforms(pixel_values)
                
                if args.pretrained_vae_model_name_or_path is not None:  # be careful with the dtype as vae seems unstable with fp16
                    pixel_values = pixel_values.to(dtype=weight_dtype)

                # Convert images to latent space
                latents = vae.encode(pixel_values.to(dtype=torch.float32)).latent_dist.sample()
                latents = latents.to(dtype=weight_dtype)
                latents = latents * vae.config.scaling_factor

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                if args.noise_offset:
                    # https://www.crosslabs.org//blog/diffusion-with-offset-noise
                    noise += args.noise_offset * torch.randn(
                        (latents.shape[0], latents.shape[1], 1, 1), device=latents.device
                    )

                bsz = latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                with torch.no_grad():
                    try:
                        time_embedding = unet.get_time_embed(sample=noisy_latents, timestep=timesteps)
                    except:
                        time_embedding = unet.module.get_time_embed(sample=noisy_latents, timestep=timesteps)

                with accelerator.autocast():
                    backbones_image = get_image_embed(brain_model_image, diffuse_prior_image, voxel_values, subj_list, accelerator, args)
                    if args.route_image:
                        backbones_image = diffuse_router_image(time_embedding, backbones_image, time_step=timesteps, total_steps=noise_scheduler.config.num_train_timesteps, context_embedding=noisy_latents, training=True)
                    prompt_embed_image = prior_image(backbones_image, diffuse_prior_image, accelerator)

                with accelerator.autocast():
                    backbones_text = get_text_embed(brain_model_text, diffuse_prior_text, voxel_values, subj_list, accelerator, args)
                    if args.route_text:
                        backbones_text = diffuse_router_text(time_embedding, backbones_text, time_step=timesteps, total_steps=noise_scheduler.config.num_train_timesteps, context_embedding=noisy_latents, training=True)
                    prompt_embed_text, pooled_prompt_embed_text = prior_text(backbones_text, diffuse_prior_text, accelerator)

                with torch.no_grad():
                    prompt_embed_text = torch.cat([sos_embed.repeat(prompt_embed_text.shape[0], 1, 1).to(accelerator.device), prompt_embed_text.repeat(1, 76, 1)], dim=1)

                with accelerator.autocast():
                    if args.model_name_text_L:
                        # Note: Text L model doesn't have router support yet
                        prompt_embed_text_L, _ = get_text_embed(brain_model_text_L, diffuse_prior_text_L, voxel_values, subj_list, accelerator, args)
                        prompt_embed_text_L = torch.cat([sos_embed_L.repeat(prompt_embed_text_L.shape[0], 1, 1).to(accelerator.device), prompt_embed_text_L.repeat(1, 76, 1)], dim=1)
                        prompt_embed_text = torch.cat((prompt_embed_text_L, prompt_embed_text), dim=-1)
                    else:
                        prompt_embed_text = torch.cat((torch.zeros(prompt_embed_text.shape[0], 77, 768).to(accelerator.device), prompt_embed_text), dim=-1)
        
                # time ids
                def compute_time_ids(original_size=args.resolution, crops_coords_top_left=0):
                    # Adapted from pipeline.StableDiffusionXLPipeline._get_add_time_ids
                    target_size = (args.resolution, args.resolution)
                    add_time_ids = list((original_size, original_size) + (crops_coords_top_left, crops_coords_top_left) + target_size)
                    add_time_ids = torch.tensor([add_time_ids])
                    add_time_ids = add_time_ids.to(accelerator.device, dtype=weight_dtype)
                    return add_time_ids
                add_time_ids = torch.cat([compute_time_ids() for _ in pixel_values])
                unet_added_conditions = {"time_ids": add_time_ids}
                unet_added_conditions.update({"text_embeds": pooled_prompt_embed_text})
                unet_added_conditions.update({"image_embeds": prompt_embed_image})
                
                # Predict the noise residual and compute loss
                model_pred = unet(noisy_latents, timesteps, prompt_embed_text, added_cond_kwargs=unet_added_conditions, return_dict=False)[0]

                # Get the target for loss depending on the prediction type
                if args.prediction_type is not None:
                    # set prediction_type of scheduler if defined
                    noise_scheduler.register_to_config(prediction_type=args.prediction_type)

                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")


                if args.snr_gamma is None:
                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                else:
                    # Compute loss-weights as per Section 3.4 of https://arxiv.org/abs/2303.09556.
                    # Since we predict the noise instead of x_0, the original formulation is slightly changed.
                    # This is discussed in Section 4.2 of the same paper.
                    snr = compute_snr(noise_scheduler, timesteps)
                    mse_loss_weights = torch.stack([snr, args.snr_gamma * torch.ones_like(timesteps)], dim=1).min(
                        dim=1
                    )[0]
                    if noise_scheduler.config.prediction_type == "epsilon":
                        mse_loss_weights = mse_loss_weights / snr
                    elif noise_scheduler.config.prediction_type == "v_prediction":
                        mse_loss_weights = mse_loss_weights / (snr + 1)

                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
                    loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
                    loss = loss.mean()

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps

                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(params_to_optimize, args.max_grad_norm)
                lr_scheduler.step()
                optimizer.step() 
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)

                if accelerator.distributed_type == DistributedType.DEEPSPEED or accelerator.is_main_process:
                    if global_step % args.checkpointing_steps == 0:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        # save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        # accelerator.save_state(save_path)
                        # logger.info(f"Saved state to {save_path}")

            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)


            if global_step >= args.max_train_steps:
                break

        if accelerator.is_main_process:
            if args.validation_prompt is not None and epoch % args.validation_epochs == 0:
                # create pipeline
                pipeline = StableDiffusionXLPipeline.from_pretrained(
                    args.pretrained_model_name_or_path,
                    vae=vae,
                    text_encoder=unwrap_model(text_encoder_one),
                    text_encoder_2=unwrap_model(text_encoder_two),
                    unet=unwrap_model(unet),
                    revision=args.revision,
                    variant=args.variant,
                    torch_dtype=weight_dtype,
                )
                images = log_validation(pipeline, args, accelerator, epoch)

                del pipeline
                torch.cuda.empty_cache()
            
            logger.info(f"Running validation for epoch {epoch}...")
            val_loss = validate_model(
                args,
                vae,
                unet,
                pixel_transforms,
                brain_model_text,
                brain_model_image,
                diffuse_prior_text,
                diffuse_prior_image,
                test_dataloader,
                noise_scheduler,
                accelerator,
                weight_dtype,
                subj_list,
                diffuse_router_image if args.route_image else None,
                diffuse_router_text if args.route_text else None,
                brain_model_text_L if args.model_name_text_L else None,
                diffuse_prior_text_L if args.model_name_text_L else None
            )
            if args.wandb_log:
                wandb.log({"train_loss": train_loss, "val_loss": val_loss, "epoch": epoch})
            
            if epoch == 0 or val_loss < best_val_loss:
                best_val_loss = val_loss
                logger.info(f"New best validation loss: {best_val_loss:.4f}")
                
                # Save the best model
                if accelerator.is_main_process:
                    unet_copy = unwrap_model(unet)
                    unet_lora_state_dict = convert_state_dict_to_diffusers(get_peft_model_state_dict(unet_copy))
                    
                    StableDiffusionXLPipeline.save_lora_weights(
                        save_directory=os.path.join(args.output_dir, "best_model"),
                        unet_lora_layers=unet_lora_state_dict,
                        text_encoder_lora_layers=None,
                        text_encoder_2_lora_layers=None,
                    )
                    
                    if args.route_image or args.route_text:
                        save_routers(accelerator, os.path.join(args.output_dir, "best_model"), "best", 
                                    diffuse_router_image if args.route_image else None, 
                                    diffuse_router_text if args.route_text else None)
                    
                    if diffuse_prior_image.requires_grad_:
                        save_prior(accelerator, os.path.join(args.output_dir, 'best_model'), "best", diffuse_prior_image, diffuse_prior_text)
                    
                    logger.info(f"Saved best model to {os.path.join(args.output_dir, 'best_model')}")
            
            # Save the final model
            unet_copy = unwrap_model(unet)
            unet_lora_state_dict = convert_state_dict_to_diffusers(get_peft_model_state_dict(unet_copy))
            
            StableDiffusionXLPipeline.save_lora_weights(
                save_directory=os.path.join(args.output_dir),
                unet_lora_layers=unet_lora_state_dict,
                text_encoder_lora_layers=None,
                text_encoder_2_lora_layers=None,
            )
            
            if args.route_image or args.route_text:
                save_routers(accelerator, os.path.join(args.output_dir), "final", 
                            diffuse_router_image if args.route_image else None, 
                            diffuse_router_text if args.route_text else None)

            if diffuse_prior_image.requires_grad_:
                save_prior(accelerator, os.path.join(args.output_dir), "final", diffuse_prior_image, diffuse_prior_text)

        if epoch == 30:
            break

    # Save the lora layers
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        if args.route_image: 
            save_routers(accelerator, args.output_dir, "final", diffuse_router_image, None)
        if args.route_text:
            save_routers(accelerator, args.output_dir, "final", None, diffuse_router_text)

        unet = unwrap_model(unet)
        unet_lora_state_dict = convert_state_dict_to_diffusers(get_peft_model_state_dict(unet))

        text_encoder_lora_layers = None
        text_encoder_2_lora_layers = None

        StableDiffusionXLPipeline.save_lora_weights(
            save_directory=args.output_dir,
            unet_lora_layers=unet_lora_state_dict,
            text_encoder_lora_layers=text_encoder_lora_layers,
            text_encoder_2_lora_layers=text_encoder_2_lora_layers,
        )

        del unet
        torch.cuda.empty_cache()

        # Final inference
        # Make sure vae.dtype is consistent with the unet.dtype
        if args.mixed_precision == "fp16":
            vae.to(weight_dtype)
        # Load previous pipeline
        pipeline = StableDiffusionXLPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            vae=vae,
            revision=args.revision,
            variant=args.variant,
            torch_dtype=weight_dtype,
        )

        # load attention processors
        pipeline.load_lora_weights(args.output_dir)

        # run inference
        if args.validation_prompt and args.num_validation_images > 0:
            images = log_validation(pipeline, args, accelerator, epoch, is_final_validation=True)

        if args.push_to_hub:
            save_model_card(
                repo_id,
                images=images,
                base_model=args.pretrained_model_name_or_path,
                dataset_name=args.dataset_name,
                train_text_encoder=args.train_text_encoder,
                repo_folder=args.output_dir,
                vae_path=args.pretrained_vae_model_name_or_path,
            )
            upload_folder(
                repo_id=repo_id,
                folder_path=args.output_dir,
                commit_message="End of training",
                ignore_patterns=["step_*", "epoch_*"],
            )

    accelerator.end_training()

@torch.no_grad()
def validate_model(
    args,
    vae,
    unet,
    pixel_transforms,
    brain_model_text,
    brain_model_image,
    diffuse_prior_text,
    diffuse_prior_image,
    test_dataloader, 
    noise_scheduler,
    accelerator,
    weight_dtype,
    subj_list,
    diffuse_router_image=None,
    diffuse_router_text=None,
    brain_model_text_L=None, 
    diffuse_prior_text_L=None
):
    logger.info("Running validation...")
    unet.eval()
    if diffuse_router_image is not None: diffuse_router_image.eval()
    if diffuse_router_text is not None: diffuse_router_text.eval()
    
    val_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch in test_dataloader:
            # Unpack batch data
            pixel_values, voxel_values, image_idx = batch['image'], batch['voxel'], batch['img_idx']
            pixel_values, voxel_values = pixel_values.to(accelerator.device), voxel_values.to(accelerator.device)
            if pixel_transforms is not None:
                pixel_values = pixel_transforms(pixel_values)

            bsz = pixel_values.shape[0]
            
            if args.pretrained_vae_model_name_or_path is not None:
                pixel_values = pixel_values.to(dtype=weight_dtype)

            # Convert images to latent space
            latents = vae.encode(pixel_values.to(dtype=torch.float32)).latent_dist.sample()
            latents = latents.to(dtype=weight_dtype)
            latents = latents * vae.config.scaling_factor

            # Sample noise
            noise = torch.randn_like(latents)
            if args.noise_offset:
                noise += args.noise_offset * torch.randn(
                    (latents.shape[0], latents.shape[1], 1, 1), device=latents.device
                )

            # Sample timesteps
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
            timesteps = timesteps.long()

            # Add noise to latents
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            with torch.no_grad():
                time_embedding = unet.module.get_time_embed(sample=noisy_latents, timestep=timesteps)
                # time_embedding = unet.module.time_embedding(time_embedding)

            # Get embeddings from brain models
            with accelerator.autocast():
                prompt_embed_image, prompt_embed_text = [], []
                for i in range(3):
                    backbones_image = get_image_embed(brain_model_image, diffuse_prior_image, [voxel_values[:, i]], subj_list[:1], accelerator, args)
                    if args.route_image:
                        backbones_image = diffuse_router_image(time_embedding, backbones_image, time_step=timesteps, total_steps=noise_scheduler.config.num_train_timesteps, context_embedding=noisy_latents, training=False)
                    prompt_embed_image_ = prior_image(backbones_image, diffuse_prior_image, accelerator)[0]
                    
                    backbones_text = get_text_embed(brain_model_text, diffuse_prior_text, [voxel_values[:, i]], subj_list[:1], accelerator, args)
                    if args.route_text:
                        backbones_text = diffuse_router_text(time_embedding, backbones_text, time_step=timesteps, total_steps=noise_scheduler.config.num_train_timesteps, context_embedding=noisy_latents, training=False)
                    prompt_embed_text_, pooled_prompt_embed_text = prior_text(backbones_text, diffuse_prior_text, accelerator)
                    prompt_embed_text_ = torch.cat([sos_embed.repeat(prompt_embed_text_.shape[0], 1, 1).to(accelerator.device), prompt_embed_text_.repeat(1, 76, 1)], dim=1)                        

                    if brain_model_text_L is not None:
                        prompt_embed_text_L, _ = get_text_embed(brain_model_text_L, diffuse_prior_text_L, [voxel_values[:, i]], subj_list[:1], accelerator, args)
                        prompt_embed_text = torch.cat((prompt_embed_text_L, prompt_embed_text), dim=-1)
                    else:
                        prompt_embed_text_ = torch.cat((torch.zeros(prompt_embed_text_.shape[0], 77, 768).to(accelerator.device), prompt_embed_text_), dim=-1)

                    prompt_embed_image.append(prompt_embed_image_)
                    prompt_embed_text.append(prompt_embed_text_)
            prompt_embed_image, prompt_embed_text = torch.stack(prompt_embed_image, 0).mean(0), torch.stack(prompt_embed_text, 0).mean(0)
            prompt_embed_image = [prompt_embed_image]


            def compute_time_ids(original_size=args.resolution, crops_coords_top_left=0):
                target_size = (args.resolution, args.resolution)
                add_time_ids = list((original_size, original_size) + (crops_coords_top_left, crops_coords_top_left) + target_size)
                add_time_ids = torch.tensor([add_time_ids])
                add_time_ids = add_time_ids.to(accelerator.device, dtype=weight_dtype)
                return add_time_ids
            
            add_time_ids = torch.cat([compute_time_ids() for _ in pixel_values])
            unet_added_conditions = {"time_ids": add_time_ids}
            unet_added_conditions.update({"text_embeds": pooled_prompt_embed_text})
            unet_added_conditions.update({"image_embeds": prompt_embed_image})
            
            # Predict noise residual
            model_pred = unet(noisy_latents, timesteps, prompt_embed_text, added_cond_kwargs=unet_added_conditions, return_dict=False)[0]

            # Get target for loss
            if args.prediction_type is not None:
                noise_scheduler.register_to_config(prediction_type=args.prediction_type)

            if noise_scheduler.config.prediction_type == "epsilon":
                target = noise
            elif noise_scheduler.config.prediction_type == "v_prediction":
                target = noise_scheduler.get_velocity(latents, noise, timesteps)
            else:
                raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

            # Compute loss
            if args.snr_gamma is None:
                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
            else:
                snr = compute_snr(noise_scheduler, timesteps)
                mse_loss_weights = torch.stack([snr, args.snr_gamma * torch.ones_like(timesteps)], dim=1).min(dim=1)[0]
                if noise_scheduler.config.prediction_type == "epsilon":
                    mse_loss_weights = mse_loss_weights / snr
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    mse_loss_weights = mse_loss_weights / (snr + 1)

                loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
                loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
                loss = loss.mean()
                
            val_loss += loss.item()
            num_batches += 1
    
    # Return the average validation loss
    avg_val_loss = val_loss / max(num_batches, 1)
    logger.info(f"Validation loss: {avg_val_loss:.4f}")
    return avg_val_loss

if __name__ == "__main__":
    main()