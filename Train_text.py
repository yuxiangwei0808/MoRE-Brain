import argparse
import gc
import json
import logging
import os
import random
import sys
from collections import OrderedDict

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import webdataset as wds
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from einops import rearrange
from tqdm import tqdm

# SDXL unCLIP requires code from https://github.com/Stability-AI/generative-models/tree/main``
sys.path.append('MindEyeV2/src/generative_models/')
sys.path.append('MindEyeV2/src')

# tf32 data type is faster than standard float32
torch.backends.cuda.matmul.allow_tf32 = True

import MindEyeV2.src.utils as utils
from brain_encoder import BrainEncoder, BrainDiffusionPriorEncoder
from brain_moe_encoder import BrainMoE, BrainMoEHier, BrainMoEMulti
from clip_encoders import CLIPImageEncoder, CLIPTextEncoderDual

# Configuration constants
DATA_TYPE = torch.float16
NUM_DEVICES = torch.cuda.device_count() if torch.cuda.is_available() else 1


def setup_distributed_training():
    """Configure distributed training with accelerate."""
    local_rank = int(os.getenv('RANK', 0))
    print("LOCAL RANK ", local_rank)
    
    # First use "accelerate config" in terminal and setup using deepspeed stage 2 with CPU offloading!
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(split_batches=False, mixed_precision="fp16", kwargs_handlers=[kwargs])
    
    return accelerator, local_rank


def configure_batch_size(batch_size, num_devices):
    """Handle batch size configuration for interactive and distributed modes."""
    if utils.is_interactive():  # set batch size here if using interactive notebook
        global_batch_size = batch_size = 128
    else:
        if "GLOBAL_BATCH_SIZE" not in os.environ: 
            os.environ["GLOBAL_BATCH_SIZE"] = '128'
        global_batch_size = os.environ["GLOBAL_BATCH_SIZE"]
        batch_size = int(os.environ["GLOBAL_BATCH_SIZE"]) // num_devices
    
    return global_batch_size, batch_size


# Initialize distributed training
accelerator, local_rank = setup_distributed_training()
global_batch_size, batch_size = configure_batch_size(128, NUM_DEVICES)  # Default batch size

print("PID of this process =", os.getpid())
device = accelerator.device
print("device:", device)
world_size = accelerator.state.num_processes
distributed = not accelerator.state.distributed_type == 'NO'
print(accelerator.state)

print(f"distributed = {distributed}, num_devices = {NUM_DEVICES}, "
      f"local rank = {local_rank}, world size = {world_size}, data_type = {DATA_TYPE}")

# Only print from the main process in distributed training
print = accelerator.print

def create_argument_parser():
    """Create and configure the argument parser for text training."""
    parser = argparse.ArgumentParser(
        description="Brain-to-Text Model Training Configuration",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Data and paths
    data_group = parser.add_argument_group('Data and Paths')
    data_group.add_argument(
        "--model_name", type=str, 
        default="singlesubject_subj01_40sess_ViT-BigG-prior-text-MoE",
        help="Name of model, used for checkpoint saving and wandb logging"
    )
    data_group.add_argument(
        "--data_path", type=str, default='data/NSD',
        help="Path to where NSD data is stored / where to download it to"
    )
    data_group.add_argument(
        "--cache_dir", type=str, default='data/NSD/.cache',
        help="Path to where misc. files downloaded from huggingface are stored"
    )
    data_group.add_argument(
        "--ckpt_path", type=str, default='',
        help="Path to checkpoint to load"
    )
    data_group.add_argument(
        "--multisubject_ckpt", type=str, default=None,
        help="Path to pre-trained multisubject model to finetune from"
    )
    
    # Subject and session configuration
    subject_group = parser.add_argument_group('Subject and Session Configuration')
    subject_group.add_argument(
        "--subj", type=int, default=1, choices=[1,2,3,4,5,6,7,8],
        help="Subject to validate on"
    )
    subject_group.add_argument(
        "--num_sessions", type=int, default=40,
        help="Number of training sessions to include"
    )
    subject_group.add_argument(
        "--multi_subject", action=argparse.BooleanOptionalAction, default=False,
        help="Whether to train on multiple subjects"
    )
    subject_group.add_argument(
        "--new_test", action=argparse.BooleanOptionalAction, default=True,
        help="Whether to use new test set"
    )
    
    # Training parameters
    training_group = parser.add_argument_group('Training Parameters')
    training_group.add_argument(
        "--batch_size", type=int, default=16,
        help="Batch size (can be increased if only training retrieval submodule)"
    )
    training_group.add_argument(
        "--num_epochs", type=int, default=150,
        help="Number of epochs of training"
    )
    training_group.add_argument(
        "--max_lr", type=float, default=3e-4,
        help="Maximum learning rate"
    )
    training_group.add_argument(
        "--lr_scheduler_type", type=str, default='cycle', choices=['cycle','linear'],
        help="Learning rate scheduler type"
    )
    training_group.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility"
    )
    training_group.add_argument(
        "--train_router_only", action=argparse.BooleanOptionalAction, default=False,
        help="Only train expert routers with frozen experts (for cross-subject); need to supply ckpt_path"
    )
    
    # Model configuration
    model_group = parser.add_argument_group('Model Configuration')
    model_group.add_argument(
        "--n_blocks", type=int, default=4,
        help="Number of transformer blocks"
    )
    model_group.add_argument(
        "--hidden_dim", type=int, default=4096,
        help="Hidden dimension size"
    )
    model_group.add_argument(
        "--b_size", type=int, default=-1,
        help="Bottleneck size"
    )
    model_group.add_argument(
        "--is_image", default=False, action=argparse.BooleanOptionalAction,
        help="Whether to use image processing"
    )
    model_group.add_argument(
        "--blurry_recon", action=argparse.BooleanOptionalAction, default=False,
        help="Whether to output blurry reconstructions"
    )
    
    # Prior configuration
    prior_group = parser.add_argument_group('Prior Configuration')
    prior_group.add_argument(
        "--use_prior", action=argparse.BooleanOptionalAction, default=False,
        help="Whether to train diffusion prior or just rely on retrieval part"
    )
    prior_group.add_argument(
        "--prior_scale", type=float, default=0,
        help="Multiply diffusion prior loss by this"
    )
    prior_group.add_argument(
        "--prior_image", action=argparse.BooleanOptionalAction, default=False,
        help="Diffusion prior toward image"
    )
    prior_group.add_argument(
        "--prior_scale_image", type=float, default=0,
        help="Multiply diffusion prior image loss by this"
    )
    prior_group.add_argument(
        "--prior_clip_scale", default=0, type=float,
        help="Prior CLIP loss scale"
    )
    
    # Loss configuration
    loss_group = parser.add_argument_group('Loss Configuration')
    loss_group.add_argument(
        "--clip_scale", type=float, default=1.,
        help="Multiply contrastive loss by this number"
    )
    loss_group.add_argument(
        "--blur_scale", type=float, default=.5,
        help="Multiply loss from blurry reconstructions by this number"
    )
    loss_group.add_argument(
        "--mse_scale", default=0., type=float,
        help="MSE loss scale"
    )
    loss_group.add_argument(
        "--cos_scale", default=0., type=float,
        help="Cosine loss scale"
    )
    loss_group.add_argument(
        "--temp_coeff", default=1., type=float,
        help="Temperature coefficient"
    )
    
    # Data augmentation
    aug_group = parser.add_argument_group('Data Augmentation')
    aug_group.add_argument(
        "--use_image_aug", action=argparse.BooleanOptionalAction, default=False,
        help="Whether to use image augmentation"
    )
    aug_group.add_argument(
        "--mixup_pct", type=float, default=.33,
        help="Proportion of training when to switch from BiMixCo to SoftCLIP"
    )
    
    # Checkpointing and logging
    checkpoint_group = parser.add_argument_group('Checkpointing and Logging')
    checkpoint_group.add_argument(
        "--ckpt_saving", action=argparse.BooleanOptionalAction, default=True,
        help="Whether to save checkpoints"
    )
    checkpoint_group.add_argument(
        "--ckpt_interval", type=int, default=5,
        help="Save backup checkpoint and reconstruct every x epochs"
    )
    checkpoint_group.add_argument(
        "--wandb_log", action=argparse.BooleanOptionalAction, default=False,
        help="Whether to log to wandb"
    )
    checkpoint_group.add_argument(
        "--wandb_project", type=str, default="stability",
        help="Wandb project name"
    )
    
    return parser


# Parse arguments and create global variables
parser = create_argument_parser()
args = parser.parse_args()

# Create global variables from arguments for backwards compatibility
for attribute_name in vars(args).keys():
    globals()[attribute_name] = getattr(args, attribute_name)
    
def setup_environment_and_augmentation():
    """Set up output directories and import augmentation libraries if needed."""
    # Set random seed for reproducibility
    utils.seed_everything(seed)
    
    # Setup output directory for checkpoints
    outdir = os.path.abspath(f'checkpoints/{model_name}')
    if not os.path.exists(outdir) and ckpt_saving:
        os.makedirs(outdir, exist_ok=True)
    
    # Import and setup augmentation if needed
    img_augment = None
    if use_image_aug or blurry_recon:
        import kornia
        from kornia.augmentation.container import AugmentationSequential
        
        if use_image_aug:
            img_augment = AugmentationSequential(
                kornia.augmentation.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1, p=0.3),
                same_on_batch=False,
                data_keys=["input"],
            )
    
    return outdir, img_augment


def configure_subjects_and_test_data():
    """Configure subject list and test dataset URLs based on training mode."""
    # Setup subject list for training
    if multi_subject:
        subj_list = np.arange(1, 9)
        subj_list = subj_list[subj_list != subj]  # Remove test subject from training
        test_subj = subj_list[0]  # Can't validate on held out person, use first in list
    else:
        subj_list = [subj]
        test_subj = subj
    
    print(f"subj_list: {subj_list}, num_sessions: {num_sessions}")
    
    # Configure test dataset based on subject and test set version
    if not new_test:  # Original test set from before full dataset released
        test_sizes = {3: 2113, 4: 1985, 6: 2113, 8: 1985}
        num_test = test_sizes.get(test_subj, 2770)
        test_url = f"{data_path}/wds/subj0{test_subj}/test/0.tar"
    else:  # Larger test set from after full dataset released
        test_sizes = {3: 2371, 4: 2188, 6: 2371, 8: 2188}
        num_test = test_sizes.get(test_subj, 3000)
        test_url = f"{data_path}/wds/subj0{test_subj}/new_test/0.tar"
    
    print(f"Test URL: {test_url}")
    
    return subj_list, test_subj, test_url, num_test


def my_split_by_node(urls): 
    """WebDataset node splitting function."""
    return urls


def create_webdataset(url, resampled=True):
    """Create a WebDataset with consistent configuration."""
    return wds.WebDataset(url, resampled=resampled, nodesplitter=my_split_by_node)\
              .shuffle(750, initial=1500, rng=random.Random(42))\
              .decode("torch")\
              .rename(behav="behav.npy", past_behav="past_behav.npy", 
                     future_behav="future_behav.npy", olds_behav="olds_behav.npy")\
              .to_tuple(*["behav", "past_behav", "future_behav", "olds_behav"])


# Initialize environment
outdir, img_augment = setup_environment_and_augmentation()
subj_list, test_subj, test_url, num_test = configure_subjects_and_test_data()

# Setup test dataset
test_data = create_webdataset(test_url, resampled=False)
test_dl = torch.utils.data.DataLoader(test_data, batch_size=num_test, shuffle=False, drop_last=True, pin_memory=True)
print(f"Loaded test dl for subj{test_subj}!\n")

# Configure training parameters
num_voxels_list = []

if multi_subject:
    nsessions_allsubj = np.array([40, 40, 32, 30, 40, 32, 40, 30])
    num_samples_per_epoch = (750 * 40) // NUM_DEVICES 
else:
    num_samples_per_epoch = (750 * num_sessions) // NUM_DEVICES 

print("Dividing batch size by subject list length for concatenation during training...") 
batch_size = batch_size // len(subj_list)
num_iterations_per_epoch = num_samples_per_epoch // (batch_size * len(subj_list))

print(f"Batch size: {batch_size}, Iterations per epoch: {num_iterations_per_epoch}, "
      f"Samples per epoch: {num_samples_per_epoch}")

# Setup training data for each subject
train_data = {}
train_dl = {}
num_voxels = {}
voxels = {}

for s in subj_list:
    print(f"Training with {num_sessions} sessions for subject {s}")
    if multi_subject:
        train_url = f"{data_path}/wds/subj0{s}/train/" + "{0.." + f"{nsessions_allsubj[s-1]-1}" + "}.tar"
    else:
        train_url = f"{data_path}/wds/subj0{s}/train/" + "{0.." + f"{num_sessions-1}" + "}.tar"
    print(f"Train URL: {train_url}")
    
    # Create training dataset
    train_set = create_webdataset(train_url, resampled=True)
    train_data[f'subj0{s}'] = train_set
    train_dl[f'subj0{s}'] = torch.utils.data.DataLoader(
        train_data[f'subj0{s}'], batch_size=batch_size, shuffle=False, 
        drop_last=False, pin_memory=True
    )

    # Load voxel data
    f = h5py.File(f'{data_path}/betas_all_subj0{s}_fp32_renorm.hdf5', 'r')
    betas = f['betas'][:]
    betas = torch.Tensor(betas).to("cpu").to(DATA_TYPE)
    num_voxels_list.append(betas[0].shape[-1])
    num_voxels[f'subj0{s}'] = betas[0].shape[-1]
    voxels[f'subj0{s}'] = betas
    print(f"num_voxels for subj0{s}: {num_voxels[f'subj0{s}']}")

print("Loaded all subj train dls and betas!\n")

# Model and CLIP configuration
arch = "ViT-bigG-14" if 'BigG' in model_name else "ViT-L-14"
version = "laion2b_s39b_b160k" if arch == "ViT-bigG-14" else "laion2b_s32b_b82k"

# Ensure blurry reconstruction is disabled for text training
assert blurry_recon == False

# Load caption data
with open(f'{data_path}/coco_captions.json', 'r') as f:
    all_gt = json.load(f)

# Configure text encoder
OUTPUT_MODE = 'BigG'
TRUNCATION = True
clip_embedder = CLIPTextEncoderDual(truncation=TRUNCATION, output_mode=OUTPUT_MODE)
# Alternative: clip_embedder = T5Constra(version='large', device=device)

# Determine embedding dimensions based on encoder type
if 'CLIPTextEncoder' in clip_embedder.__class__.__name__ and OUTPUT_MODE == 'joint':
    clip_emb_dim = 2048
elif clip_embedder.__class__.__name__ == 'T5Constra':
    clip_emb_dim = 1024
else:
    clip_emb_dim = 1280 if 'BigG' in model_name else 768

# Determine sequence dimension based on encoder configuration
if ((clip_embedder.__class__.__name__ == 'CLIPTextEncoderDual' and TRUNCATION) or 
    clip_embedder.__class__.__name__ == 'T5Constra' or 'pool' in OUTPUT_MODE):
    clip_seq_dim = 1
else:
    clip_seq_dim = 77

clip_embedder = clip_embedder.to(device)

# Setup image encoder for prior training if needed
if use_prior and prior_image:
    clip_image_embedder = CLIPImageEncoder(version='BigG', is_proj=True, return_type='pooled')
    gt_images = h5py.File(f'{data_path}/coco_images_224_float16.hdf5', 'r')['images']
    clip_seq_dim_target = 1
    diffusion_prior_image = BrainDiffusionPriorEncoder(
        clip_emb_dim, depth=6, dim_head=52, heads=clip_emb_dim//52, 
        clip_seq_dim=clip_seq_dim, clip_seq_dim_target=clip_seq_dim_target, timesteps=100
    )

if blurry_recon:
    from diffusers import AutoencoderKL    
    autoenc = AutoencoderKL(
        down_block_types=['DownEncoderBlock2D', 'DownEncoderBlock2D', 'DownEncoderBlock2D', 'DownEncoderBlock2D'],
        up_block_types=['UpDecoderBlock2D', 'UpDecoderBlock2D', 'UpDecoderBlock2D', 'UpDecoderBlock2D'],
        block_out_channels=[128, 256, 512, 512],
        layers_per_block=2,
        sample_size=256,
    )
    ckpt = torch.load(f'{cache_dir}/sd_image_var_autoenc.pth')
    autoenc.load_state_dict(ckpt)
    
    autoenc.eval()
    autoenc.requires_grad_(False)
    autoenc.to(device)
    utils.count_params(autoenc)
    
    from autoencoder.convnext import ConvnextXL
    cnx = ConvnextXL(f'{cache_dir}/convnext_xlarge_alpha0.75_fullckpt.pth')
    cnx.requires_grad_(False)
    cnx.eval()
    cnx.to(device)
    
    mean = torch.tensor([0.485, 0.456, 0.406]).to(device).reshape(1,3,1,1)
    std = torch.tensor([0.228, 0.224, 0.225]).to(device).reshape(1,3,1,1)
    
    blur_augs = AugmentationSequential(
        kornia.augmentation.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1, p=0.8),
        kornia.augmentation.RandomGrayscale(p=0.1),
        kornia.augmentation.RandomSolarize(p=0.1),
        kornia.augmentation.RandomResizedCrop((224,224), scale=(.9,.9), ratio=(1,1), p=1.0),
        data_keys=["input"],
    )

# model = BrainEncoder(num_voxels_list, hidden_dim, blurry_recon, n_blocks, clip_emb_dim, clip_seq_dim, clip_scale, interm_out=False, enc_version='v1')
model = BrainMoEMulti(num_voxels_list, hidden_dim, blurry_recon, n_blocks, clip_emb_dim, clip_seq_dim, clip_scale, interm_out=False, enc_version='v1', num_exp_0=2, capacity_factor_0=1, num_exp_layer=3, exp_factor_list=[2, 2, 2], cap_fac_list=[1, 1, 1], meta=False, b_size=b_size)
if use_prior:    
    diffusion_prior = BrainDiffusionPriorEncoder(clip_emb_dim, depth=6, dim_head=52, heads=clip_emb_dim//52, clip_seq_dim=clip_seq_dim, timesteps=100)

if ckpt_path:
    print(f"Loading ckpt from {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    model_ckpt = OrderedDict({k: v for k, v in checkpoint['model_state_dict'].items() if 'ridge' not in k})
    model.load_state_dict(model_ckpt, strict=False)
    if use_prior:
        diffusion_prior.load_state_dict(checkpoint['prior_state_dict'])
    del checkpoint
    print("Loaded ckpt!")

no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']

if train_router_only:
    assert ckpt_path, "Must provide ckpt_path to train router only!"
    freeze_patterns = ['expert_list', 'expert_0', 'c_proj_list', 'clip_proj_0', 'b_proj_list', 'backbone_proj_0']
    frozen_params = 0
    total_params = 0
    for name, param in model.named_parameters():
        total_params += param.numel()
        if any(pattern in name for pattern in freeze_patterns):
            param.requires_grad = False
            frozen_params += param.numel()
            print(f"Freezing parameter: {name}")
    print(f"Frozen {frozen_params}/{total_params} parameters ({frozen_params/total_params:.2%})")

opt_grouped_parameters = [
    {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay) and p.requires_grad], 'weight_decay': 1e-2},
    {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay) and p.requires_grad], 'weight_decay': 0.0},
]

if use_prior:
    opt_grouped_parameters.extend([
        {'params': [p for n, p in diffusion_prior.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 1e-2},
        {'params': [p for n, p in diffusion_prior.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ])
    if prior_image:
        opt_grouped_parameters.extend([
            {'params': [p for n, p in diffusion_prior_image.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 1e-2},
            {'params': [p for n, p in diffusion_prior_image.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ])

optimizer = torch.optim.AdamW(opt_grouped_parameters, lr=max_lr)

if lr_scheduler_type == 'linear':
    lr_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        total_iters=int(np.floor(num_epochs * num_iterations_per_epoch)),
        last_epoch=-1
    )
elif lr_scheduler_type == 'cycle':
    total_steps=int(np.floor(num_epochs * num_iterations_per_epoch))
    print("total_steps", total_steps)
    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=max_lr,
        total_steps=total_steps,
        final_div_factor=1000,
        last_epoch=-1, pct_start=2/num_epochs
    )
    
def save_ckpt(tag):
    ckpt_path = outdir+f'/{tag}.pth'
    if accelerator.is_main_process:
        unwrapped_model = accelerator.unwrap_model(model)
        saving_dict = {
            'epoch': epoch,
            'model_state_dict': unwrapped_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'train_losses': losses,
            'test_losses': test_losses,
            'lrs': lrs,
            }
        if use_prior: 
            saving_dict.update({'prior_state_dict': accelerator.unwrap_model(diffusion_prior).state_dict()})
            if prior_image: 
                saving_dict.update({'prior_image_state_dict': accelerator.unwrap_model(diffusion_prior_image).state_dict()})
        torch.save(saving_dict, ckpt_path)
    print(f"\n---saved {outdir}/{tag} ckpt!---\n")

def load_ckpt(tag,load_lr=True,load_optimizer=True,load_epoch=True,strict=True,outdir=outdir,multisubj_loading=False): 
    print(f"\n---loading {outdir}/{tag}.pth ckpt---\n")
    checkpoint = torch.load(outdir+'/last.pth', map_location='cpu')
    state_dict = checkpoint['model_state_dict']
    if multisubj_loading: # remove incompatible ridge layer that will otherwise error
        state_dict.pop('ridge.linears.0.weight',None)
    model.load_state_dict(state_dict, strict=strict)
    if load_epoch:
        globals()["epoch"] = checkpoint['epoch']
        print("Epoch",epoch)
    if load_optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if load_lr:
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
    del checkpoint

logging.info(f"Done with model preparations on rank-{device}!")
num_params = utils.count_params(model)

if local_rank==0 and wandb_log: # only use main process for wandb logging
    import wandb
    wandb_project = 'brainGen_text-finetune'
    print(f"wandb {wandb_project} run {model_name}")
    # need to configure wandb beforehand in terminal with "wandb init"!
    wandb_config = vars(args)
    print("wandb_config:\n",wandb_config)
    print("wandb_id:",model_name)
    wandb.init(
        project=wandb_project,
        name=model_name,
        config=wandb_config,
        # id=model_name,
        # resume="allow",
        # settings=wandb.Settings(init_timeout=120)
    )
    wandb.run.log_code(".", include_fn=lambda path: path.endswith(".py"))
else:
    wandb_log = False


epoch = 0
losses, test_losses, lrs = [], [], []
best_test_loss = 1e9
torch.cuda.empty_cache()

# load multisubject stage1 ckpt if set
if multisubject_ckpt is not None:
    load_ckpt("last", outdir=multisubject_ckpt, load_lr=False, load_optimizer=False, load_epoch=False,strict=False,multisubj_loading=True)

NUM_CAPTIONS_PER_IMAGE = 5
train_dls = [train_dl[f'subj0{s}'] for s in subj_list]
# pre-loading all test data
test_gts, test_voxels, test_images = [], [], []
for test_i, (behav, past_behav, future_behav, old_behav) in enumerate(test_dl):
    assert test_i < 1
    test_gt, test_voxel, test_image = [], [], []
    # all test samples should be loaded per batch such that test_i should never exceed 0
    assert len(behav) == num_test

    ## Average same-image repeats ##
    voxel = voxels[f'subj0{subj}'][behav[:,0,5].cpu().long()].unsqueeze(1)
    
    image = behav[:,0,0].cpu().long()

    unique_image, sort_indices = torch.unique(image, return_inverse=True)
    for im in unique_image:
        locs = torch.where(im == image)[0]
        if len(locs)==1:
            locs = locs.repeat(3)
        elif len(locs)==2:
            locs = locs.repeat(2)[:3]
        assert len(locs)==3
        locs = locs[:1]
        
        if is_image:
            test_gt.append(torch.Tensor(all_gt[im]))
        else:
            test_gt.append(all_gt[str(int(im))][:NUM_CAPTIONS_PER_IMAGE])
            if use_prior and prior_image: test_image.append(torch.Tensor(gt_images[im]))
        test_voxel.append(voxel[locs])

    if is_image:
        test_gt = torch.stack(test_gt, 0)
    elif use_prior and prior_image:
        test_image = torch.stack(test_image, 0)
    test_voxel = torch.stack(test_voxel, 0)

    test_indices = torch.arange(len(test_voxel))[:300]
    voxel = test_voxel[test_indices]
    if is_image: 
        gt = test_gt[test_indices]
    else: 
        gt = [test_gt[i] for i in test_indices]
        if use_prior and prior_image:
            image = test_image[test_indices]
            test_images.append(image)

    assert len(gt) == 300
    test_gts.append(gt)
    test_voxels.append(voxel)

# leaving out test_dl since we will only have local_rank 0 device do evals
model, optimizer, *train_dls, lr_scheduler = accelerator.prepare(model, optimizer, *train_dls, lr_scheduler)
if use_prior:
    diffusion_prior = accelerator.prepare(diffusion_prior)
    if prior_image: 
        diffusion_prior_image = accelerator.prepare(diffusion_prior_image)
        clip_image_embedder = accelerator.prepare(clip_image_embedder)
if blurry_recon:
    autoenc, cnx = accelerator.prepare(autoenc, cnx)
accelerator.print(f"device: {accelerator.device}")

print(f"{model_name} starting with epoch {epoch} / {num_epochs}")
progress_bar = tqdm(range(epoch, num_epochs), ncols=1200, disable=(local_rank!=0))

mse = nn.MSELoss()
l1 = nn.L1Loss()
best_mse, best_acc = 1e9, 0
soft_loss_temps = utils.cosine_anneal(0.004 * temp_coeff, 0.0075 * temp_coeff, num_epochs - int(mixup_pct * num_epochs))

# pre-load all batches for this epoch (it's MUCH faster to pre-load in bulk than to separate loading per batch)
voxel_iters = {} # empty dict because diff subjects have differing # of voxels
if is_image:
    gt_iters = torch.zeros(num_iterations_per_epoch, batch_size*len(subj_list), 3, 224, 224).float()
else:
    gt_iters = {}
    if use_prior: image_iters = torch.zeros(num_iterations_per_epoch, batch_size*len(subj_list), 3, 224, 224).float()
annot_iters = {}
perm_iters, betas_iters, select_iters = {}, {}, {}

# logging.info(f'Begin epoch {epoch} on-{local_rank}')
for s, train_dl in enumerate(train_dls):
    with accelerator.autocast():
        iter = -1
        for behav0, past_behav0, future_behav0, old_behav0 in train_dl:
            # Load images to cpu from hdf5 (requires sorted indexing)
            # or load captions
            image_idx = behav0[:, 0, 0].cpu().long().numpy()
            
            iter += 1

            if iter in gt_iters: 
                gt_iters[iter].extend([all_gt[str(i)][:NUM_CAPTIONS_PER_IMAGE] for i in image_idx])
            else: 
                gt_iters[iter] = [all_gt[str(i)][:NUM_CAPTIONS_PER_IMAGE] for i in image_idx]

            if use_prior and prior_image:
                image0 = torch.stack([torch.tensor(gt_images[i], dtype=DATA_TYPE) for i in image_idx])
                image_iters[iter,s*batch_size:s*batch_size+batch_size] = image0
            
            # Load voxels for current batch, matching above indexing
            voxel_idx = behav0[:, 0, 5].cpu().long().numpy()
            voxel0 = voxels[f'subj0{subj_list[s]}'][voxel_idx]
            voxel0 = torch.Tensor(voxel0).unsqueeze(1)

            if epoch < int(mixup_pct * num_epochs):
                voxel0, perm, betas, select = utils.mixco(voxel0)
                perm_iters[f"subj0{subj_list[s]}_iter{iter}"] = perm
                betas_iters[f"subj0{subj_list[s]}_iter{iter}"] = betas
                select_iters[f"subj0{subj_list[s]}_iter{iter}"] = select

            voxel_iters[f"subj0{subj_list[s]}_iter{iter}"] = voxel0

            if iter >= num_iterations_per_epoch-1:
                break


for epoch in progress_bar:
    model.train()

    fwd_percent_correct = 0.
    bwd_percent_correct = 0.
    test_fwd_percent_correct = 0.
    test_bwd_percent_correct = 0.

    fwd_percent_correct_prior = 0.
    bwd_percent_correct_prior = 0.
    test_fwd_percent_correct_prior = 0.
    test_bwd_percent_correct_prior = 0.
    
    recon_cossim = 0.
    test_recon_cossim = 0.
    recon_mse = 0.
    test_recon_mse = 0.

    loss_clip_total = 0.
    loss_blurry_total = 0.
    loss_blurry_cont_total = 0.
    test_loss_clip_total = 0.
    
    loss_prior_total = 0.
    test_loss_prior_total = 0.

    loss_prior_image_total = 0.
    test_loss_prior_image_total = 0.

    mse_voxel = 0.
    test_mse_voxel = 0.

    cos_voxel = 0.
    test_cos_voxel = 0.

    blurry_pixcorr = 0.
    test_blurry_pixcorr = 0. # needs >.456 to beat low-level subj01 results in mindeye v1

    # logging.info(f'Begin train on-{local_rank}')
    for train_i in range(num_iterations_per_epoch):
        with accelerator.autocast():
            optimizer.zero_grad()
            loss = 0.

            voxel_list = [voxel_iters[f"subj0{s}_iter{train_i}"].detach().to(device) for s in subj_list]

            gt = gt_iters[train_i]
            if is_image:
                gt = gt.detach().to(device)

            if use_image_aug:
                assert is_image
                gt = img_augment(gt)

            clip_target = [clip_embedder(x) for x in gt]
            # clip_target = [clip_embedder.embed(x) for x in gt]

            clip_target = torch.stack(clip_target, dim=0).permute(1, 0, 2, 3)
            # clip_target = clip_target[0]  ### only use one GT
            clip_target = clip_target.mean(0)
            assert not torch.any(torch.isnan(clip_target[0]))

            if epoch < int(mixup_pct * num_epochs):
                perm_list = [perm_iters[f"subj0{s}_iter{train_i}"].detach().to(device) for s in subj_list]
                perm = torch.cat(perm_list, dim=0)
                betas_list = [betas_iters[f"subj0{s}_iter{train_i}"].detach().to(device) for s in subj_list]
                betas = torch.cat(betas_list, dim=0)
                select_list = [select_iters[f"subj0{s}_iter{train_i}"].detach().to(device) for s in subj_list]
                select = torch.cat(select_list, dim=0)

            if 'MoE' in model_name:
                backbones, clip_voxels, blurry_image_enc_, lb_loss = model(voxel_list, subj_list, training=True)
                loss += lb_loss * 1
            else:
                backbones, clip_voxels, blurry_image_enc_ = model(voxel_list, subj_list)

            if use_prior:
                # project from text to image embeddings as standard diffusion prior
                if prior_image: # prior from text emb (1*dim) to image emb (1*dim)
                    with torch.no_grad():
                        clip_target_image = clip_image_embedder(image_iters[train_i].detach().to(device))
                    
                    if isinstance(backbones, list) or backbones.ndim == 4:
                        loss_prior_image = sum([diffusion_prior_image(text_embed=backbones[i], image_embed=clip_target_image) for i in range(len(backbones))])
                        loss_prior_image = loss_prior_image / len(backbones)
                    else:
                        loss_prior_image, _ = diffusion_prior_image(text_embed=backbones, image_embed=clip_target_image)
                    loss += (loss_prior_image * prior_scale_image)

                    loss_prior_image = accelerator.gather(loss_prior_image.detach()).mean().item()
                    loss_prior_image_total += loss_prior_image

                if isinstance(backbones, list) or backbones.ndim == 4:
                    prior_out = []
                    loss_prior = 0.
                    for b in range(len(backbones)):
                        l, o = diffusion_prior(text_embed=backbones[b], image_embed=clip_target)
                        loss_prior += l
                        prior_out.append(o)
                    prior_out = torch.stack(prior_out, dim=0)
                    loss_prior /= len(clip_voxels)
                else:
                    loss_prior, prior_out = diffusion_prior(text_embed=backbones, image_embed=clip_target)
         
                if isinstance(prior_out, list) or prior_out.ndim == 4:
                    recon_cossim_single = np.mean([nn.functional.cosine_similarity(prior_out[i], clip_target).mean().item() for i in range(len(prior_out))])
                    recon_mse_single = np.mean([mse(prior_out[i], clip_target).item() for i in range(len(prior_out))])
                else:
                    recon_cossim_single = nn.functional.cosine_similarity(prior_out, clip_target).mean().item()
                    recon_mse_single = mse(prior_out, clip_target).item()
                
                loss += (loss_prior * prior_scale)
                loss_prior = accelerator.gather(loss_prior.detach()).mean().item()
                loss_prior_total += loss_prior
            else:
                prior_out = clip_voxels
                recon_cossim_single = nn.functional.cosine_similarity(clip_voxels, clip_target).mean().item()
                recon_mse_single = mse(clip_voxels, clip_target[i]).item()

            recon_cossim += recon_cossim_single
            recon_mse += recon_mse_single

            clip_voxels_norm = nn.functional.normalize(clip_voxels.flatten(-2), dim=-1)
            clip_target_norm = nn.functional.normalize(clip_target.flatten(-2), dim=-1)
            if clip_scale > 0:            
                if epoch < int(mixup_pct * num_epochs):
                    if isinstance(backbones, list) or backbones.ndim == 4:
                        loss_clip = sum([utils.mixco_nce(clip_voxels_norm[i], clip_target_norm,
                            temp=.006 * temp_coeff, perm=perm, betas=betas, select=select) for i in range(len(backbones))])
                        loss_clip = loss_clip / len(backbones)
                    else:
                        loss_clip = utils.mixco_nce(clip_voxels_norm, clip_target_norm, temp=.006 * temp_coeff, perm=perm, betas=betas, select=select)
                else:
                    epoch_temp = soft_loss_temps[epoch-int(mixup_pct*num_epochs)]
                    if isinstance(backbones, list) or backbones.ndim == 4:
                        loss_clip = sum([utils.soft_clip_loss(clip_voxels_norm[i], clip_target_norm, temp=epoch_temp) for i in range(len(backbones))])
                        loss_clip = loss_clip / len(backbones)
                    else:
                        loss_clip = utils.soft_clip_loss(clip_voxels_norm, clip_target_norm, temp=epoch_temp)
                        
                loss += (loss_clip * clip_scale)
                loss_clip = accelerator.gather(loss_clip.detach()).mean().item()
                loss_clip_total += loss_clip

                # forward and backward top 1 accuracy        
                if isinstance(backbones, list) or backbones.ndim == 4:
                    labels = torch.arange(backbones.shape[1]).to(backbones.device)
                    fwd_percent_correct += np.mean([utils.topk(utils.batchwise_cosine_similarity(clip_voxels_norm[i], clip_target_norm), labels, k=1).item() for i in range(len(backbones))])
                    bwd_percent_correct += np.mean([utils.topk(utils.batchwise_cosine_similarity(clip_target_norm, clip_voxels_norm[i]), labels, k=1).item() for i in range(len(backbones))])
                    fwd_percent_correct_prior += np.mean([utils.topk(utils.batchwise_cosine_similarity(prior_out[i].flatten(-2), clip_target_norm), labels, k=1).item() for i in range(len(backbones))])
                    bwd_percent_correct_prior += np.mean([utils.topk(utils.batchwise_cosine_similarity(clip_target_norm, prior_out[i].flatten(-2)), labels, k=1).item() for i in range(len(backbones))])           
                else:
                    labels = torch.arange(len(clip_voxels_norm)).to(clip_voxels_norm.device)
                    fwd_percent_correct += utils.topk(utils.batchwise_cosine_similarity(clip_voxels_norm, clip_target_norm), labels, k=1).item()
                    bwd_percent_correct += utils.topk(utils.batchwise_cosine_similarity(clip_target_norm, clip_voxels_norm), labels, k=1).item()

                    fwd_percent_correct_prior += utils.topk(utils.batchwise_cosine_similarity(prior_out.flatten(-2), clip_target.flatten(-2)), labels, k=1).item()
                    bwd_percent_correct_prior += utils.topk(utils.batchwise_cosine_similarity(clip_target.flatten(-2), prior_out.flatten(-2)), labels, k=1).item()

            ### Prior CLip ###
            if prior_clip_scale > 0:
                clip_prior_norm = nn.functional.normalize(prior_out.flatten(-2), dim=-1)
                if epoch < int(mixup_pct * num_epochs):
                    if isinstance(backbones, list) or backbones.ndim == 4:
                        loss_clip = sum([utils.mixco_nce(clip_prior_norm[i], clip_target_norm, temp=.006 * temp_coeff, perm=perm, betas=betas, select=select) for i in range(len(backbones))])
                        loss_clip = loss_clip / len(backbones)
                    else:
                        loss_clip = utils.mixco_nce(clip_prior_norm[i], clip_target_norm, temp=.006 * temp_coeff, perm=perm, betas=betas, select=select)
                else:
                    epoch_temp = soft_loss_temps[epoch-int(mixup_pct*num_epochs)]
                    if isinstance(backbones, list) or backbones.ndim == 4:
                        loss_clip = sum([utils.soft_clip_loss(clip_prior_norm[i], clip_target_norm, temp=epoch_temp) for i in range(len(backbones))])
                        loss_clip = loss_clip / len(backbones)
                    else:
                        loss_clip = utils.soft_clip_loss(clip_prior_norm, clip_target_norm, temp=epoch_temp)
                        
                loss += (loss_clip * prior_clip_scale)
                loss_clip = accelerator.gather(loss_clip.detach()).mean().item()
                loss_clip_total += loss_clip
            ### Prior CLip ###
            
            if mse_scale > 0:
                if isinstance(backbones, list) or backbones.ndim == 4:
                    loss_mse = sum([mse(clip_voxels[i], clip_target) for i in range(len(backbones))])
                    loss_mse = loss_mse / len(backbones)
                else:
                    loss_mse = mse(clip_voxels, clip_target)
                loss += (loss_mse * mse_scale)
                mse_voxel += loss_mse.item()

            if cos_scale > 0:
                clip_voxels_norm = F.normalize(prior_out.flatten(-2), dim=-1)
                if isinstance(backbones, list) or backbones.ndim == 4:
                    loss_cos = sum([1 - F.cosine_similarity(clip_voxels_norm[i], clip_target_norm).mean() for i in range(len(backbones))])
                    loss_cos /= len(backbones)
                else:
                    loss_cos = 1 - F.cosine_similarity(clip_voxels_norm, clip_target_norm).mean()
                loss += (loss_cos * cos_scale)

            utils.check_loss(loss)
            accelerator.backward(loss)
            optimizer.step()

            loss = accelerator.gather(loss.detach()).mean().item()
            losses.append(loss)
            lrs.append(optimizer.param_groups[0]['lr'])

            if lr_scheduler_type is not None:
                lr_scheduler.step()

    fwd_percent_correct = accelerator.gather(torch.tensor(fwd_percent_correct, device=accelerator.device)).mean().item()
    bwd_percent_correct = accelerator.gather(torch.tensor(bwd_percent_correct, device=accelerator.device)).mean().item()
    fwd_percent_correct_prior = accelerator.gather(torch.tensor(fwd_percent_correct_prior, device=accelerator.device)).mean().item()
    bwd_percent_correct_prior = accelerator.gather(torch.tensor(bwd_percent_correct_prior, device=accelerator.device)).mean().item()
    recon_cossim = accelerator.gather(torch.tensor(recon_cossim, device=accelerator.device)).mean().item()
    recon_mse = accelerator.gather(torch.tensor(recon_mse, device=accelerator.device)).mean().item()
    mse_voxel = accelerator.gather(torch.tensor(mse_voxel, device=accelerator.device)).mean().item()
    cos_voxel = accelerator.gather(torch.tensor(cos_voxel, device=accelerator.device)).mean().item()

    # logging.info(f'Begin evaluation on rank-{local_rank}')
    torch.cuda.empty_cache()
    gc.collect()
    model.eval()
    
    TEST_BZ = 64
    with torch.no_grad(), accelerator.autocast():
        loss = 0.
        for test_gt_, test_voxel_ in zip(test_gts, test_voxels):
            for test_i, i in enumerate(range(0, len(test_gt_), TEST_BZ)): 
                test_voxel = test_voxel_[i:i+TEST_BZ]
                test_gt = test_gt_[i:i+TEST_BZ]                    
                voxel = test_voxel.to(device)
                if is_image: test_gt = test_gt.to(device)
                
                if not is_image and isinstance(gt, list):
                    clip_target = [clip_embedder(x) for x in test_gt]
                    # clip_target = [clip_embedder.embed(x) for x in test_gt]

                    clip_target = torch.stack(clip_target, dim=0).permute(1, 0, 2, 3)
                    clip_target = clip_target[0]  ### only use one GT
                else:
                    clip_target, pooled_target = clip_embedder(test_gt)

                for rep in range(voxel.shape[1]):
                    if 'MoE' in model_name:
                        backbone0, clip_voxels0, blurry_image_enc_, lb_loss = model([voxel[:, rep]], subj_list[:1], training=False)
                        loss += lb_loss * 1
                    else:
                        backbone0, clip_voxels0, blurry_image_enc_ = model([voxel[:, rep]], subj_list[:1])

                    if rep == 0:
                        clip_voxels = clip_voxels0
                        backbones = backbone0
                    else:
                        clip_voxels += clip_voxels0
                        backbones += backbone0
                clip_voxels /= 3
                backbones /= 3

                if use_prior:
                    if prior_image:
                        with torch.no_grad():
                            clip_target_image = clip_image_embedder(test_images[0][i:i+TEST_BZ].detach().to(device))
                        if isinstance(backbones, list) or backbones.ndim == 4:
                            loss_prior_image = sum([diffusion_prior_image(text_embed=backbones[i], image_embed=clip_target_image) for i in range(len(backbones))])
                            loss_prior_image = loss_prior_image / len(backbones)
                        else:
                            loss_prior_image, _ = diffusion_prior_image(text_embed=backbones, image_embed=clip_target_image)
                        loss += (loss_prior_image * prior_scale_image)
                        test_loss_prior_image_total += loss_prior_image.item()

                    if isinstance(backbones, list) or backbones.ndim == 4:
                        prior_out = []
                        for b in range(len(backbones)):
                            l, o = diffusion_prior(text_embed=backbones[b], image_embed=clip_target)
                            loss_prior += l
                            prior_out.append(o)
                        loss_prior /= len(backbones)
                        prior_out = torch.stack(prior_out, dim=0)
                        test_recon_cossim += np.mean([nn.functional.cosine_similarity(prior_out[i], clip_target).mean().item() for i in range(len(backbones))])
                        test_recon_mse += np.mean([mse(prior_out[i], clip_target).item() for i in range(len(backbones))])
                    else:
                        loss_prior, prior_out = diffusion_prior(text_embed=backbones, image_embed=clip_target)
                        test_recon_cossim += nn.functional.cosine_similarity(prior_out, clip_target).mean().item()
                        test_recon_mse += mse(prior_out, clip_target).item()
                    
                    loss += (loss_prior * prior_scale)
                    loss_prior_total += loss_prior
                else:  # text reconstruction target
                    prior_out = clip_voxels
                    test_recon_cossim += np.mean([nn.functional.cosine_similarity(clip_voxels, clip_target[i]).mean().item() for i in range(len(clip_target))])
                    test_recon_mse += np.mean([mse(clip_voxels, clip_target[i]).item() for i in range(len(clip_target))])

                clip_voxels_norm = nn.functional.normalize(clip_voxels.flatten(-2), dim=-1)
                clip_target_norm = nn.functional.normalize(clip_target.flatten(-2), dim=-1)

                if clip_scale > 0:
                    if isinstance(backbones, list) or backbones.ndim == 4:
                        loss_clip = sum([utils.mixco_nce(clip_voxels_norm[i], clip_target_norm, temp=.006 * temp_coeff) for i in range(len(backbones))])
                        loss_clip = loss_clip / len(backbones)
                    else:
                        loss_clip = utils.soft_clip_loss(clip_voxels_norm, clip_target_norm, temp=.006 * temp_coeff)

                    test_loss_clip_total += loss_clip.item()
                    loss += loss_clip * clip_scale

                    # forward and backward top 1 accuracy
                    if isinstance(backbones, list) or backbones.ndim == 4:
                        labels = torch.arange(backbones.shape[1]).to(backbones.device)
                        test_fwd_percent_correct += np.mean([utils.topk(utils.batchwise_cosine_similarity(clip_voxels_norm[i], clip_target_norm), labels, k=1).item() for i in range(len(backbones))])
                        test_bwd_percent_correct += np.mean([utils.topk(utils.batchwise_cosine_similarity(clip_target_norm, clip_voxels_norm[i]), labels, k=1).item() for i in range(len(backbones))])
                        
                        test_fwd_percent_correct_prior += np.mean([utils.topk(utils.batchwise_cosine_similarity(prior_out[i], clip_target_norm), labels, k=1).item() for i in range(len(backbones))])
                        test_bwd_percent_correct_prior += np.mean([utils.topk(utils.batchwise_cosine_similarity(clip_target_norm, prior_out[i]), labels, k=1).item() for i in range(len(backbones))])
                    else:
                        labels = torch.arange(len(clip_voxels_norm)).to(clip_voxels_norm.device) 
                        test_fwd_percent_correct += utils.topk(utils.batchwise_cosine_similarity(clip_voxels_norm, clip_target_norm), labels, k=1).item()
                        test_bwd_percent_correct += utils.topk(utils.batchwise_cosine_similarity(clip_target_norm, clip_voxels_norm), labels, k=1).item()
                        
                        test_fwd_percent_correct_prior += utils.topk(utils.batchwise_cosine_similarity(prior_out.flatten(-2), clip_target.flatten(-2)), labels, k=1).item()
                        test_bwd_percent_correct_prior += utils.topk(utils.batchwise_cosine_similarity(clip_target.flatten(-2), prior_out.flatten(-2)), labels, k=1).item()

                if mse_scale > 0:
                    if isinstance(backbones, list) or backbones.ndim == 4:
                        loss_mse = sum([mse(clip_voxels[i], clip_target) for i in range(len(backbones))])
                        loss_mse = loss_mse / len(backbones)
                    else:
                        loss_mse = mse(clip_voxels, clip_target)
                    loss += (loss_mse * mse_scale)
                    test_mse_voxel += loss_mse.item()

                if cos_scale > 0:
                    clip_voxels_norm = F.normalize(prior_out.flatten(-2), dim=-1)
                    if isinstance(backbones, list) or backbones.ndim == 4:
                        loss_cos = sum([1 - F.cosine_similarity(clip_voxels_norm[i], clip_target_norm).mean() for i in range(len(backbones))])
                        loss_cos /= len(backbones)
                    else:
                        loss_cos = 1 - F.cosine_similarity(clip_voxels_norm, clip_target_norm).mean()
                    loss += (loss_cos * cos_scale)

        if local_rank == 0:
            utils.check_loss(loss)  
            test_losses.append(loss.item())

            logs = {"train/loss": np.mean(losses[-(train_i+1):]),
                "test/loss": np.mean(test_losses[-(test_i+1):]),
                "train/lr": lrs[-1],
                "train/num_steps": len(losses),
                "test/num_steps": len(test_losses),
                "train/fwd_pct_correct": fwd_percent_correct / (train_i + 1),
                "train/bwd_pct_correct": bwd_percent_correct / (train_i + 1),
                "test/test_fwd_pct_correct": test_fwd_percent_correct / (test_i + 1),
                "test/test_bwd_pct_correct": test_bwd_percent_correct / (test_i + 1),
                "train/loss_clip_total": loss_clip_total / (train_i + 1),
                "train/loss_blurry_total": loss_blurry_total / (train_i + 1),
                "train/loss_blurry_cont_total": loss_blurry_cont_total / (train_i + 1),
                "test/loss_clip_total": test_loss_clip_total / (test_i + 1),
                "train/blurry_pixcorr": blurry_pixcorr / (train_i + 1),
                "test/blurry_pixcorr": test_blurry_pixcorr / (test_i + 1),
                "train/recon_cossim": recon_cossim / (train_i + 1),
                "test/recon_cossim": test_recon_cossim / (test_i + 1),
                "train/recon_mse": recon_mse / (train_i + 1),
                "test/recon_mse": test_recon_mse / (test_i + 1),
                "train/loss_prior": loss_prior_total / (train_i + 1),
                "test/loss_prior": test_loss_prior_total / (test_i + 1),
                "train/mse_voxel": mse_voxel / (train_i + 1),
                "test/mse_voxel": test_mse_voxel / (test_i + 1),
                "train/cos_voxel": cos_voxel / (train_i + 1),
                "test/cos_voxel": test_cos_voxel / (test_i + 1),
                "train/fwd_pct_correct_prior": fwd_percent_correct_prior / (train_i + 1),
                "train/bwd_pct_correct_prior": bwd_percent_correct_prior / (train_i + 1),
                "test/test_fwd_pct_correct_prior": test_fwd_percent_correct_prior / (test_i + 1),
                "test/test_bwd_pct_correct_prior": test_bwd_percent_correct_prior / (test_i + 1),
                }

            # if finished training, save jpg recons if they exist
            if (epoch == num_epochs-1) or (epoch % ckpt_interval == 0):
                if blurry_recon:    
                    image_enc = autoenc.encode(2*test_gt[:4]-1).latent_dist.mode() * 0.18215
                    # transform blurry recon latents to images and plot it
                    fig, axes = plt.subplots(1, 8, figsize=(10, 4))
                    jj=-1
                    for j in [0,1,2,3]:
                        jj+=1
                        axes[jj].imshow(utils.torch_to_Image((autoenc.decode(image_enc[[j]]/0.18215).sample / 2 + 0.5).clamp(0,1)))
                        axes[jj].axis('off')
                        jj+=1
                        axes[jj].imshow(utils.torch_to_Image((autoenc.decode(image_enc_pred[[j]]/0.18215).sample / 2 + 0.5).clamp(0,1)))
                        axes[jj].axis('off')

                    if wandb_log:
                        logs[f"test/blur_recons"] = wandb.test_gt(fig, caption=f"epoch{epoch:03d}")
                        plt.close()
                    else:
                        plt.show()

            if wandb_log: wandb.log(logs)
            logs.update({'model_name': model_name})
            progress_bar.set_postfix(**logs)
    
    # logging.info(f"Finished evaluation on rank-{local_rank}")
    # Save model checkpoint and reconstruct
    if local_rank == 0:
        # if ckpt_saving: save_ckpt(f'last')
        if (test_bwd_percent_correct_prior + test_fwd_percent_correct_prior) / 2 > best_acc:
            best_acc = (test_bwd_percent_correct_prior + test_fwd_percent_correct_prior) / 2
            # if ckpt_saving: save_ckpt('best_acc')
        if test_recon_mse < best_mse:
            best_mse = test_recon_mse
            if ckpt_saving: save_ckpt('best_mse')

    # wait for other GPUs to catch up if needed
    accelerator.wait_for_everyone()
    torch.cuda.empty_cache()

print("\n===Finished!===\n")
# if ckpt_saving:
#     save_ckpt(f'last')


plt.plot(losses)
plt.show()
plt.plot(test_losses)
plt.show()
