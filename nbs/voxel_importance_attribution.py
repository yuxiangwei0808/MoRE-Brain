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
from collections import OrderedDict
import nibabel as nib

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision import transforms
from accelerate import Accelerator
from transformers import CLIPImageProcessor
from diffusers import AutoPipelineForText2Image

from arr2nib import arr2nib

sys.path.append('MindEyeV2/src/generative_models/')
sys.path.append('MindEyeV2/src')

from clip_encoders import CLIPImageEncoder
from brain_encoder import BrainEncoder, BrainDiffusionPriorEncoder
from brain_moe_encoder import BrainMoE, BrainMoEMulti, BrainMoEHier
from beta_encoders import *
from load_data import load_nsd, load_train_data
from routers import *

from captum.attr import IntegratedGradients, GradientShap, LayerGradientShap, LayerIntegratedGradients, NoiseTunnel, NeuronGradientShap, NeuronIntegratedGradients
from captum.attr import visualization as viz

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def arr2nib(arr, affine=None, header=None, filename=None):
    """
    Convert a numpy array to a NIfTI image and save it to a file.

    Parameters
    ----------
    arr : numpy.ndarray
        The input array to be converted to a NIfTI image.
    affine : numpy.ndarray, optional
        The affine transformation matrix. If None, an identity matrix is used.
    header : nibabel.Nifti1Header, optional
        The header for the NIfTI image. If None, a default header is created.
    filename : str, optional
        The filename to save the NIfTI image. If None, the image is not saved.

    Returns
    -------
    nibabel.Nifti1Image
        The NIfTI image created from the input array.
    """
    
    if affine is None:
        affine = np.eye(4)
    
    # Create a default header if none is provided
    if header is None:
        header = nib.Nifti1Header()
    
    # Create the NIfTI image
    img = nib.Nifti1Image(arr, affine, header)
    
    # Save the image if a filename is provided
    if filename:
        nib.save(img, filename)
    
    return img
    

def find_most_important_expert(info):
    expert_id = None
    for i, layer in enumerate(['layer_0', 'layer_1', 'layer_2', 'layer_3']):
        weights = info[layer].mean(-1)  # averging over feature dimension
        if expert_id is None:
            expert_id = weights.argmax(0)
        else:
            expert_selections = torch.stack((expert_id * 2, expert_id * 2 + 1), dim=0).to(weights.device)
            weights = torch.gather(weights, 0, expert_selections)
            expert_id = weights.argmax(0)
            expert_id = torch.gather(expert_selections, 0, expert_id.unsqueeze(0)).squeeze(0)
    return expert_id

class BrainInterpreter:
    def __init__(self, model, prior, diffuse_router, clip_embedder, subj_list, device='cuda'):
        """
        Initialize the brain interpreter with your trained model
        
        Args:
            model: Your trained BrainGen model
            clip_embedder: CLIP embedder used during training
            subj_list: List of subjects used in the model
            device: Device to run the analysis on
        """
        self.model = model
        self.prior = prior
        self.diffuse_router = diffuse_router
        self.clip_embedder = clip_embedder
        self.subj_list = subj_list
        self.device = device
        
        # Ensure model is in eval mode
        self.model.eval()
        
        # Create Captum attribution methods
        self.integrated_gradients = IntegratedGradients(self.get_model_output)
        self.gradient_shap = GradientShap(self.get_model_output)
        self.noise_tunnel = NoiseTunnel(self.integrated_gradients)
        
    def get_model_output(self, voxel_tensor, routing=False, get_exp_out=False, level_idx=None, exp_idx=None):
        """
        Wrapper function for model to work with Captum
        Returns CLIP embedding from voxel data for a single subject
        """
        # Reshape input as needed
        if voxel_tensor.dim() == 2:  # Batch x Voxels
            voxel_tensor = voxel_tensor.unsqueeze(1)  # Add channel dimension
            
        # Put into list for single subject
        voxel_list = [voxel_tensor.to(self.device)]
        
        # Forward pass through model, only use the first subject
        if 'MoE' in self.model.__class__.__name__:
            backbones, clip_voxels, _, _ = self.model(voxel_list, self.subj_list[:1], training=False, return_exp_out=get_exp_out)
        else:
            backbones, clip_voxels, _ = self.model(voxel_list, self.subj_list[:1])

        # route before
        if get_exp_out:
            if routing:
                backbones = self.diffuse_router(backbones, training=False)
                _, backbones = self.prior(text_embed=backbones, image_embed=torch.zeros_like(backbones).to(backbones.device))
            else:
                assert isinstance(backbones, list)
                backbones = backbones[level_idx][exp_idx]
                _, backbones = self.prior(text_embed=backbones, image_embed=torch.zeros_like(backbones).to(backbones.device))
        else:
            _, backbones = self.prior(text_embed=backbones, image_embed=torch.zeros_like(backbones).to(backbones.device))
            
        return backbones, clip_voxels
    
    def compute_integrated_gradients(self, voxel_tensor, target_embedding, n_steps=50):
        """
        Compute Integrated Gradients attribution for a given voxel tensor
        
        Args:
            voxel_tensor: Input voxel tensor (already preprocessed)
            target_embedding: Target CLIP embedding to compare against
            n_steps: Number of steps for integral approximation
            
        Returns:
            Attribution tensor of same shape as input voxel tensor
        """
        all_attributions = []
        for i in range(voxel_tensor.shape[0]):
            # Extract single sample
            single_voxel = voxel_tensor[i:i+1]
            single_target = target_embedding[i:i+1]
            
            # Create a wrapper that computes cosine similarity with target embedding
            def wrapped_model(input_tensor):
                backbones, clip_voxels = self.get_model_output(input_tensor)
                backbone_norm = nn.functional.normalize(backbones, dim=-1)
                output_norm = nn.functional.normalize(clip_voxels, dim=-1)
                target_norm = nn.functional.normalize(single_target, dim=-1)
                similarity = ((output_norm * target_norm).sum(dim=-1) + (backbone_norm * target_norm).sum(dim=-1)) / 2
                return similarity
            
            specific_ig = IntegratedGradients(wrapped_model)
            
            # Set requires_grad for input
            single_voxel.requires_grad = True
            
            # Create baseline (zero tensor)
            baseline = torch.zeros_like(single_voxel)
            
            # Compute attribution - target=None because our wrapper already computes the similarity
            attr = specific_ig.attribute(
                single_voxel,
                baselines=baseline,
                target=None,  # No need for target, as the wrapper returns similarity scores
                n_steps=n_steps,
                method="gausslegendre",
                return_convergence_delta=False
            )
            
            all_attributions.append(attr)
        
        attributions = torch.cat(all_attributions, dim=0)
        return attributions
    
    def compute_gradient_shap(self, voxel_tensor, target_embedding, n_samples=50, stdevs=0.01):
        """
        Compute GradientSHAP attribution for a given voxel tensor
        
        Args:
            voxel_tensor: Input voxel tensor
            target_embedding: Target CLIP embedding
            n_samples: Number of samples for SHAP
            stdevs: Standard deviation for noise
            
        Returns:
            Attribution tensor of same shape as input voxel tensor
        """
        all_attributions = []
        for i in range(voxel_tensor.shape[0]):
            single_voxel = voxel_tensor[i:i+1]
            single_target = target_embedding[i:i+1]
            
            def wrapped_model(input_tensor):
                backbones, clip_voxels = self.get_model_output(input_tensor)
                backbone_norm = nn.functional.normalize(backbones, dim=-1)
                output_norm = nn.functional.normalize(clip_voxels, dim=-1)
                target_norm = nn.functional.normalize(single_target, dim=-1)
                similarity = ((output_norm * target_norm).sum(dim=-1) + (backbone_norm * target_norm).sum(dim=-1)) / 2
                return similarity
            
            specific_shap = GradientShap(wrapped_model)
            single_voxel.requires_grad = True
            baseline = torch.zeros_like(single_voxel)

            attr = specific_shap.attribute(
                single_voxel,
                baselines=baseline,
                target=None,  # No need for target
                n_samples=n_samples,
                stdevs=stdevs
            )            
            all_attributions.append(attr)
        
        attributions = torch.cat(all_attributions, dim=0)
        return attributions

    def similarity_wrapper(self, voxel_tensor, target_embedding_norm, **kwargs):
        backbone_out, clip_out = self.get_model_output(voxel_tensor)

        # Normalize outputs
        backbone_norm = F.normalize(backbone_out, dim=-1)
        clip_norm = F.normalize(clip_out, dim=-1)

        # Calculate similarity (average of backbone and clip similarity to target)
        similarity = ((clip_norm * target_embedding_norm).sum(dim=-1) + \
                      (backbone_norm * target_embedding_norm).sum(dim=-1)) / 2
        return similarity

    def similarity_wrapper_mask(self, voxel_tensor, target_embedding_norm, target_level, target_exp):
        backbone_out, clip_out = self.get_model_output(voxel_tensor, get_exp_out=True, level_idx=target_level, exp_idx=target_exp)
        
        # Normalize outputs
        backbone_norm = F.normalize(backbone_out, dim=-1)
        clip_norm = F.normalize(clip_out, dim=-1)

        # Calculate similarity (average of backbone and clip similarity to target)
        similarity = ((clip_norm * target_embedding_norm).sum(dim=-1) + \
                      (backbone_norm * target_embedding_norm).sum(dim=-1)) / 2
        return similarity

    def compute_layer_attribution(self, voxel_tensor, target_embedding, target_layer, method='ig', mask=False, layer_idx=None, exp_idx=None, **kwargs):
        """
        Compute attribution for a specific layer's output. Use Neuron-based method

        Args:
            voxel_tensor: Input voxel tensor (B, N_voxels)
            target_embedding: Target CLIP embedding (B, EmbDim)
            target_layer: The specific nn.Module layer instance within the model to attribute.
            method: 'ig' for LayerIntegratedGradients or 'shap' for LayerGradientShap.
            **kwargs: Additional arguments for the Captum method (n_steps, n_samples, etc.)

        Returns:
            Attribution tensor with shape matching the output of target_layer.
        """
        def aggregate_expert_output_selector(output_tensor):
            # output_tensor shape is likely [batch_size, seq_dim, emb_dim]
            # We sum across sequence and embedding dimensions for the first batch item.
            if output_tensor.ndim > 0: # Handle potential scalar outputs if they occur
                return torch.sum(output_tensor[0])
            return output_tensor # Return as is if scalar

        if mask:
            sim_wrapper = self.similarity_wrapper_mask
        else:
            sim_wrapper = self.similarity_wrapper

        all_attributions = []
        target_embedding_norm = F.normalize(target_embedding.to(self.device), dim=-1)

        # Ensure models are on the correct device
        self.model.to(self.device)
        self.prior.to(self.device)

        for i in range(voxel_tensor.shape[0]):
            single_voxel = voxel_tensor[i:i+1].to(self.device) # Input to the *overall* model
            single_voxel.requires_grad = True # Input requires grad for gradient methods
            baseline = torch.zeros_like(single_voxel) # Baseline matches model input

            if method == 'ig':
                layer_attributor = NeuronIntegratedGradients(
                    partial(sim_wrapper, target_embedding_norm=target_embedding_norm[i:i+1], target_level=layer_idx, target_exp=exp_idx),
                    target_layer
                )
                attribute_args = {'n_steps': kwargs.get('n_steps', 50), 'method': "gausslegendre"}
            elif method == 'shap':
                layer_attributor = NeuronGradientShap(
                    partial(sim_wrapper, target_embedding_norm=target_embedding_norm[i:i+1], target_level=layer_idx, target_exp=exp_idx),
                    target_layer
                )
                attribute_args = {'n_samples': kwargs.get('n_samples', 50), 'stdevs': kwargs.get('stdevs', 0.01)}
            else:
                raise ValueError("Unsupported layer attribution method")

            # Set baseline distribution for GradientSHAP if needed
            if method == 'shap':
                 n_samples = attribute_args['n_samples']
                 stdevs = attribute_args['stdevs']
                 baseline_dist = torch.randn(n_samples, *single_voxel.shape[1:], device=self.device) * stdevs
                 attribute_args['baselines'] = baseline_dist
            else:
                 attribute_args['baselines'] = baseline # Baseline for IG

            # Compute layer attribution
            attr = layer_attributor.attribute(
                single_voxel,
                neuron_selector=aggregate_expert_output_selector,
                **attribute_args
            )
            # Layer attr shape depends on layer output. Detach and move to CPU.
            all_attributions.append(attr.detach().cpu())

        # Concatenate results from individual samples
        # Note: Stacking might be better if layer output has multiple dimensions (B, H, W)
        attributions = torch.cat(all_attributions, dim=0)

        return attributions

    def _compute_path_scores(self, tree_dict):
        path_sums = {}
        tree = tree_dict.keys()
        tree_set = set(tree)
        
        # Identify all leaf nodes (nodes with no children)
        leaf_nodes = []
        for l, i in tree:
            left_child = (l + 1, 2 * i)
            right_child = (l + 1, 2 * i + 1)
            
            if left_child not in tree_set and right_child not in tree_set:
                leaf_nodes.append((l, i))
        
        # Find paths from root to each leaf node
        paths = []
        for leaf in leaf_nodes:
            path = [leaf]
            current = leaf
            
            while current != (0, 0):  # While not reaching the root
                parent_layer = current[0] - 1
                parent_idx = current[1] // 2
                parent = (parent_layer, parent_idx)
                
                if parent in tree_set:  # Ensure the parent exists
                    path.insert(0, parent)
                else:
                    break  # If parent doesn't exist, this isn't a valid path
                    
                current = parent
            
            if path[0] == (0, 0):  # Ensure the path starts from the root
                paths.append(path)

        for path in paths:
            s = 0
            for node in path:
                s += tree_dict[node]
            path_sums[node] = s

        return path_sums
    
    def compute_path_attribution(self, voxel_tensor, target_embedding, layer_list, method='ig', **kwargs):
        """Find the path that contributes to the output of the model."""
        def aggregate_expert_output_selector(output_tensor):
            # output_tensor shape is likely [batch_size, seq_dim, emb_dim]
            # We sum across sequence and embedding dimensions for the first batch item.
            if output_tensor.ndim > 0: # Handle potential scalar outputs if they occur
                return torch.sum(output_tensor[0])
            return output_tensor # Return as is if scalar

        sim_wrapper = self.similarity_wrapper

        all_attributions = []
        target_embedding_norm = F.normalize(target_embedding.to(self.device), dim=-1)

        # Ensure models are on the correct device
        self.model.to(self.device)
        self.prior.to(self.device)

        for i in range(voxel_tensor.shape[0]):
            expert_attrs = {(0, 0): 0}

            single_voxel = voxel_tensor[i:i+1].to(self.device) # Input to the *overall* model
            single_voxel.requires_grad = True # Input requires grad for gradient methods
            baseline = torch.zeros_like(single_voxel) # Baseline matches model input

            for target_layer, layer_id, exp_id in layer_list:
                if method == 'ig':
                    layer_attributor = NeuronIntegratedGradients(
                        partial(sim_wrapper, target_embedding_norm=target_embedding_norm[i:i+1]),
                        target_layer
                    )
                    attribute_args = {'n_steps': kwargs.get('n_steps', 50), 'method': "gausslegendre"}
                elif method == 'shap':
                    layer_attributor = NeuronGradientShap(
                        partial(sim_wrapper, target_embedding_norm=target_embedding_norm[i:i+1]),
                        target_layer
                    )
                    attribute_args = {'n_samples': kwargs.get('n_samples', 50), 'stdevs': kwargs.get('stdevs', 0.01)}
                else:
                    raise ValueError("Unsupported layer attribution method")

                # Set baseline distribution for GradientSHAP if needed
                if method == 'shap':
                    n_samples = attribute_args['n_samples']
                    stdevs = attribute_args['stdevs']
                    baseline_dist = torch.randn(n_samples, *single_voxel.shape[1:], device=self.device) * stdevs
                    attribute_args['baselines'] = baseline_dist
                else:
                    attribute_args['baselines'] = baseline # Baseline for IG

                # Compute layer attribution
                attr = layer_attributor.attribute(
                    single_voxel,
                    neuron_selector=aggregate_expert_output_selector,
                    **attribute_args
                )
                expert_attrs.update({(layer_id + 1, exp_id): attr.detach().cpu().abs().sum()})

            # expert_attrs = self._compute_path_scores(expert_attrs)
            # expert_attrs = torch.stack(list(expert_attrs.values()))  # store as an tensor (len 16)
            
            info = {'layer_0': [], 'layer_1': [], 'layer_2': [], 'layer_3': []}
            for (layer_id, exp_id), value in expert_attrs.items():
                if layer_id == 0: continue
                info[f'layer_{layer_id - 1}'].append(value)
            expert_attrs = {k: torch.stack(v)[:, None, None] for k, v in info.items()}
            expert_attrs = find_most_important_expert(expert_attrs)
            
            all_attributions.append(expert_attrs)

        all_attributions = torch.stack(all_attributions, dim=0)  # Shape: (batch_size, num_experts)
        return all_attributions 


def load_model_and_data(model_name, lora_ckpt, subj=1, load_image=True):
    num_sessions = [40, 40, 32, 30, 40, 32, 40, 30]
    num_sessions = num_sessions[subj-1]
    batch_size = 16
    train_dls, voxels, batch_size, test_dataloader, num_voxels_list, subj_list, num_iterations_per_epoch = load_nsd(num_sessions, subj, 1, batch_size, batch_size, False)
    
    # Initialize CLIP embedder
    if load_image:
        clip_embedder = CLIPImageEncoder('BigG', is_proj=True, return_type='pooled', process_img=True).to(device)
    else:
        clip_embedder = CLIPTextEncoderDual(truncation=True, output_mode='BigG')
    
    checkpoint = torch.load(f'checkpoints/{model_name}/best_mse.pth', map_location='cpu')
    model = BrainMoEMulti(num_voxels_list, 4096, False, 4, 1280, 1, 1, interm_out=False, enc_version='v1', num_exp_0=2, capacity_factor_0=1, num_exp_layer=3, exp_factor_list=[2, 2, 2], cap_fac_list=[1, 1, 1], meta=False)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model = model.to(device)
    model.eval()

    diffusion_prior = BrainDiffusionPriorEncoder(1280, depth=6, dim_head=52, heads=1280//52, clip_seq_dim=1, timesteps=100)
    diffusion_prior.load_state_dict(checkpoint['prior_state_dict'])
    diffusion_prior = diffusion_prior.to(device)
    diffusion_prior.eval()

    diffuse_router = DiffuseRouter(
        time_router_class=DiffuseTimeRouterAttn,
        space_router_class=DiffuseSpaceRouterSelfAttn,
        num_granularity_levels=4,
        num_experts_per_granularity=[2, 4, 8, 16],
        enable_time=True,  # we do not have time embedding here
        soft_time_routing=True,
    )
    if load_image:
        diffuse_router_ckpt = torch.load(f'checkpoints/sdxl-finetuned-lora/{lora_ckpt}/best_model/best_diffuse_router_image.pth', map_location='cpu')
    else:
        diffuse_router_ckpt = torch.load(f'checkpoints/sdxl-finetuned-lora/{lora_ckpt}/best_model/best_diffuse_router_text.pth', map_location='cpu')
    diffuse_router.load_state_dict(diffuse_router_ckpt)
    diffuse_router.requires_grad_(False)
    diffuse_router.to(device).eval()
    
    return model, diffusion_prior, diffuse_router, clip_embedder, test_dataloader, subj_list

def recon_volume(attrs, mask, fill_value=0):
    batch_size = attrs.shape[0]
    reconstructed = torch.full((batch_size, *mask.shape), fill_value, dtype=attrs.dtype, device=attrs.device)
    
    mask_indices = torch.nonzero(torch.tensor(mask), as_tuple=True)
    for i in range(batch_size):
        batch_indices = (torch.tensor([i]), *mask_indices)
        reconstructed[batch_indices] = attrs[i]
    
    return reconstructed.to(torch.float32).cpu().detach()

if __name__ == "__main__":
    subj = 1
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_name = f'single_subj{subj}_40sess_ViT-BigG-img-prior-noblur-4096-BrainMoEMulti2-2L3F2--all+'
    lora_ckpt = 'MoEMulti2-noMeta-SpaceRouterSelfAttn-TimeRouterAttn_temp1.5_soft-routeBefore--all+'

    mask = nib.load(f'data/NSD/ROI/subj0{subj}_nsdgeneral.nii.gz')
    affine = mask.affine
    mask = mask.get_fdata()
    mask = np.where(mask > 0, 1, 0)

    attr_scheme = 'path_top1'
    output_dir = f'Viz/subj0{subj}/{attr_scheme}'

    model, diffusion_prior, diffuse_router, clip_embedder, test_dataloader, subj_list = load_model_and_data(model_name, lora_ckpt, subj=subj, load_image=True)
    interpreter = BrainInterpreter(model, diffusion_prior, diffuse_router, clip_embedder, subj_list, device=device)


    ig_attrs, shap_attrs = [], []
    if attr_scheme == 'layerwise' or 'path' in attr_scheme:
        layers_to_analyze = OrderedDict()
        # Use unique names that can be used by the hook
        for i in range(model.num_exp_0):
            layers_to_analyze[f'exp0_bproj_{i}'] = (model.backbone_proj_0[i], 0, i)
            # layers_to_analyze[f'exp0_cproj_{i}'] = (model.clip_proj_0[i], 0, i) # Can add clip proj too

        for level_idx in range(model.num_exp_layer):
                num_experts_in_level = len(model.expert_list[level_idx])
                for expert_idx_in_level in range(num_experts_in_level):
                    lname = f'level{level_idx+1}_bproj_{expert_idx_in_level}'
                    layers_to_analyze[lname] = (model.b_proj_list[level_idx][expert_idx_in_level], level_idx+1, expert_idx_in_level)
                    # layers_to_analyze[lname] = (model.c_proj_list[level_idx][expert_idx_in_level], level_idx+1, expert_idx_in_level)
        print(f"Layers selected for analysis: {list(layers_to_analyze.keys())}")
        projected_layer_ig_results = {name: [] for name in layers_to_analyze}
        projected_layer_shap_results = {name: [] for name in layers_to_analyze}


    for i, batch in enumerate(test_dataloader):
        print(f"Processing batch {i}...")
        
        # Get voxel tensor and target image
        voxel_tensor = batch['voxel'].to(device).to(torch.float32)
        assert voxel_tensor.shape[1] == 3

        target_image = batch['image'].to(device).to(torch.float32)
        target_idx = batch['img_idx'].cpu().numpy()
        # Get target CLIP embedding
        with torch.no_grad():
            target_embedding = clip_embedder(target_image)
        
        ig, shap = [], []
        if attr_scheme == 'layerwise':
            ig_layer, shap_layer = {name: [] for name in layers_to_analyze.keys()}, {name: [] for name in layers_to_analyze.keys()}
        for i in range(3):
            if attr_scheme == 'overall':
                # Compute the overall attributions
                print("Computing Integrated Gradients...")
                ig_attributions = interpreter.compute_integrated_gradients(
                    voxel_tensor[:, i], target_embedding,
                )
                ig_attributions = recon_volume(ig_attributions, mask)
                
                print("Computing GradientSHAP...")
                shap_attributions = interpreter.compute_gradient_shap(
                    voxel_tensor[:, i], target_embedding
                )
                shap_attributions = recon_volume(shap_attributions, mask)
                ig.append(ig_attributions)
                shap.append(shap_attributions)
            elif attr_scheme == 'layerwise':
                ### Compute layer attributions
                for name, (layer, layer_idx, exp_idx) in tqdm(layers_to_analyze.items(), desc="Layer Projection", leave=False):
                    proj_layer_ig = interpreter.compute_layer_attribution(
                            voxel_tensor=voxel_tensor[:, i],
                            target_embedding=target_embedding,
                            target_layer=layer,
                            method='ig',
                            n_steps=50,
                            mask=True,
                            layer_idx=layer_idx,
                            exp_idx=exp_idx,
                        )
                    ig_layer[name].append(proj_layer_ig)
                    proj_layer_shap = interpreter.compute_layer_attribution(
                            voxel_tensor=voxel_tensor[:, i],
                            target_embedding=target_embedding,
                            target_layer=layer,
                            method='shap',
                                n_samples=50,
                                stdevs=0.01,
                                mask=True,
                                layer_idx=layer_idx,
                                exp_idx=exp_idx,
                            )
                    shap_layer[name].append(proj_layer_shap)
            elif 'path' in attr_scheme:
                ig_path_attr = interpreter.compute_path_attribution(
                    voxel_tensor=voxel_tensor[:, i],
                    target_embedding=target_embedding,
                    layer_list=list(layers_to_analyze.values()),
                    method='ig',
                    n_steps=50,
                )
                shap_path_attr = interpreter.compute_path_attribution(
                    voxel_tensor=voxel_tensor[:, i],
                    target_embedding=target_embedding,
                    layer_list=list(layers_to_analyze.values()),
                    method='shap',
                    n_samples=50,
                )
                ig.append(ig_path_attr)
                shap.append(shap_path_attr)
        
        if attr_scheme == 'overall' or attr_scheme == 'path':
            ### Overall attributions | Path attributions
            ig_attributions = torch.stack(ig, dim=0).mean(0)
            shap_attributions = torch.stack(shap, dim=0).mean(0)
            ig_attrs.append(ig_attributions)
            shap_attrs.append(shap_attributions)
        elif attr_scheme == 'layerwise':
            ### Layerwise attributions
            for name, layer in layers_to_analyze.items():
                ig_layer[name] = torch.stack(ig_layer[name], dim=0).mean(0)
                shap_layer[name] = torch.stack(shap_layer[name], dim=0).mean(0)
                projected_layer_ig_results[name].append(ig_layer[name])
                projected_layer_shap_results[name].append(shap_layer[name])
        elif attr_scheme == 'path_top1':
            ### Find the top 1 path by iterating sequentially along each level; no averaging
            ig_attrs.append(torch.stack(ig, dim=0))
            shap_attrs.append(torch.stack(shap, dim=0))

        # Visualize attributions
        # print("Visualizing results...")
        # ig_top_indices, ig_attrs = interpreter.visualize_attributions(
        #     ig_attributions, voxel_tensor, subj_list[0], "IntegratedGradients", output_dir
        # )
        
        # shap_top_indices, shap_attrs = interpreter.visualize_attributions(
        #     shap_attributions, voxel_tensor, subj_list[0], "GradientSHAP", output_dir
        # )

    os.makedirs(output_dir, exist_ok=True)

    if attr_scheme == 'overall':
        # Save overall attributions
        ig_attrs = torch.cat(ig_attrs, dim=0)
        shap_attrs = torch.cat(shap_attrs, dim=0)
        ig_attrs, shap_attrs = ig_attrs.permute(1, 2, 3, 0), shap_attrs.permute(1, 2, 3, 0)
        arr2nib(ig_attrs.numpy(), affine=affine, filename=os.path.join(output_dir, 'ig_attrs.nii.gz'))
        arr2nib(shap_attrs.numpy(), affine=affine, filename=os.path.join(output_dir, 'shap_attrs.nii.gz'))
        print('Saved attributions to disk.')
    elif attr_scheme == 'layerwise':
        ### Save layerwise attributions
        for name, attributions in projected_layer_ig_results.items():
            attributions = torch.cat(attributions, dim=0)
            attributions = recon_volume(attributions, mask).numpy()
            attributions = np.transpose(attributions, (1, 2, 3, 0))
            attributions = arr2nib(attributions, affine=affine, filename=os.path.join(output_dir, f'ig_attrs_{name}.nii.gz'))
            print(f'Saved IG attributions for {name} to disk.')
        for name, attributions in projected_layer_shap_results.items():
            attributions = torch.cat(attributions, dim=0)
            attributions = recon_volume(attributions, mask).numpy()
            attributions = np.transpose(attributions, (1, 2, 3, 0))
            attributions = arr2nib(attributions, affine=affine, filename=os.path.join(output_dir, f'shap_attrs_{name}.nii.gz'))
            print(f'Saved SHAP attributions for {name} to disk.')
    elif attr_scheme == 'path':
        ig_attrs = torch.cat(ig_attrs, dim=0)
        shap_attrs = torch.cat(shap_attrs, dim=0)
        np.save(os.path.join(output_dir, 'ig_attrs_path.npy'), ig_attrs.numpy())
        np.save(os.path.join(output_dir, 'shap_attrs_path.npy'), shap_attrs.numpy())
    elif attr_scheme == 'path_top1':
        np.save(os.path.join(output_dir, 'ig_attrs_path_top1.npy'), torch.cat(ig_attrs, dim=1).numpy())
        np.save(os.path.join(output_dir, 'shap_attrs_path_top1.npy'), torch.cat(shap_attrs, dim=1).numpy())