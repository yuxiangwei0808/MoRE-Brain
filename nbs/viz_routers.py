import os
import nibabel as nib
from tqdm import tqdm
import torch
from einops import rearrange
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy.stats import entropy
from diffusers import UNet2DConditionModel, AutoencoderKL

from voxel_importance_attribution import load_model_and_data, recon_volume

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
            expert_id = torch.gather(expert_selections, 0, expert_id.unsqueeze(0)).squeeze()
    return expert_id

def analyze_MoE_routing_patterns(model, dataloader, device):
    """
    Analyze how inputs are routed through the MoE layers
    """
    model.eval()
    all_routing_info = []
    
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            voxel_tensor = batch['voxel'].to(device).to(torch.float32)
            assert voxel_tensor.shape[1] == 3

            batch_info = []
            for j in range(3):
                vox = voxel_tensor[:, j]
                if vox.dim() == 2:  # Batch x Voxels
                    vox = vox.unsqueeze(1)  # Add channel dimension
            
                _, _, _, _, routing_info = model([vox], [1], training=False, return_exp_out=False, capture_routing=True)
                routing_info = {k: v['weights'] for k, v in routing_info.items()}
                batch_info.append(routing_info)
            batch_info = {k: torch.stack([v[k] for v in batch_info], dim=0).mean(0) for k in batch_info[0].keys()}
            top1_paths = find_most_important_expert(batch_info)
            all_routing_info.append(top1_paths)

    # all_routing_info = {k: torch.cat([v[k] for v in all_routing_info], dim=1) for k in all_routing_info[0].keys()}
    all_routing_info = torch.cat(all_routing_info, dim=0)
    
    np.save(f'{output_dir}/path/routing_info.npy', all_routing_info.cpu().numpy())


def attribute_ridge(model, dataloader, device, target_layer_idx):
    model.eval()
    num_voxels = 15724
    bz = 16
    hidden_dim = 4096

    exp_dict = {0: 2, 1: 4, 2: 8, 3: 16}
    num_exp = exp_dict[target_layer_idx]

    total_abs_grads = []
    for i, batch in tqdm(enumerate(dataloader)):
        voxel_tensor = batch['voxel'].to(device).to(torch.float32)
        assert voxel_tensor.shape[1] == 3

        total_abs_grads_per_expert = torch.zeros(bz, num_exp, num_voxels, device=device)
        for j in range(3):
            model.zero_grad()
            vox = voxel_tensor[:, j]
            if vox.dim() == 2:  # Batch x Voxels
                vox = vox.unsqueeze(1)  # Add channel dimension

            vox_input = vox.detach().clone()
            vox_input.requires_grad_(True)
            vox_ridge = model.ridge(vox_input, 0).squeeze()  # B H

            with torch.no_grad(): 
                _, _, _, _, routing_info = model([vox.detach()], [1], training=False, return_exp_out=False, capture_routing=True)
                topk_indices_layer0 = routing_info[f'layer_{target_layer_idx}']['topk_indices']
                B = topk_indices_layer0.shape[1]

            for e in range(num_exp): # Loop through each expert
                for b in range(B): # Loop through each sample in the batch
                    selected_hidden_indices = topk_indices_layer0[e, b, :] # Indices for expert e, sample b

                    assert selected_hidden_indices.numel()

                    for hidden_idx in selected_hidden_indices:
                        if hidden_idx.item() >= vox_ridge.shape[1]:
                            raise Exception(f"Warning: Selected hidden index {hidden_idx.item()} out of bounds for ridge output dim {vox_ridge.shape[1]}. Skipping.")

                        # Target activation value
                        target_activation = vox_ridge[b, hidden_idx.item()]

                        # Calculate gradient: d(target_activation) / d(vox_input[b])
                        # Use autograd.grad for efficiency and control
                        # grad_outputs=torch.ones_like(...) needed for non-scalar output backprop
                        input_grads = torch.autograd.grad(
                            outputs=target_activation,
                            inputs=vox_input, # Calculate grad w.r.t the whole batch input...
                            grad_outputs=torch.ones_like(target_activation),
                            retain_graph=True # Keep graph intact for next iteration
                        )[0].squeeze()

                        # Accumulate the absolute gradient for the specific sample 'b'
                        if input_grads is not None:
                            # We only want the gradient corresponding to sample 'b'
                            sample_grad = input_grads[b]
                            total_abs_grads_per_expert[b, e] += torch.abs(sample_grad)
                        else:
                            raise Exception(f"Warning: Got None gradient for expert {e}, sample {b}, hidden_idx {hidden_idx.item()}")

            # Detach vox_ridge to free graph memory before next view/batch
            vox_ridge = vox_ridge.detach()
            # Explicitly clear grads on vox_input before next view/batch
            if vox_input.grad is not None:
                vox_input.grad.zero_()

        total_abs_grads_per_expert /= 3
        total_abs_grads_per_expert = rearrange(total_abs_grads_per_expert.cpu(), 'b e n -> (b e) n')
        total_abs_grads_per_expert = recon_volume(total_abs_grads_per_expert)
        total_abs_grads_per_expert = rearrange(total_abs_grads_per_expert, '(b e) x y z -> b e x y z', e=num_exp)
        total_abs_grads.append(total_abs_grads_per_expert)

    total_abs_grads = torch.cat(total_abs_grads, dim=0)  # B E N
    np.save(f'{output_dir}/ridge_grads_mask-layer{target_layer_idx}.npy', total_abs_grads.numpy())


def analyze_time_router(router, dataloader, device, num_bins=20):

    def plot_router_preferences(
        timesteps, 
        attention_weights, 
        bin_centers, 
        binned_weights,
        num_granularity_levels,
        save_path
    ):
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12), sharex=True)
        
        # Use a consistent colormap for different granularity levels
        colors = plt.cm.viridis(np.linspace(0, 1, num_granularity_levels))
        
        # Plot 1: Raw attention weights with scatter points
        for i in range(num_granularity_levels):
            label = f'Level {i}' + (' (Coarsest)' if i == 0 else ' (Finest)' if i == num_granularity_levels-1 else '')
            ax1.scatter(
                timesteps, 
                attention_weights[:, i], 
                s=10,
                color=colors[i],
                alpha=0.5,
                label=label
            )
        
        ax1.set_ylabel('Attention Weight', fontsize=14)
        ax1.set_title('Raw Router Attention Weights Across Diffusion Process', fontsize=16)
        ax1.legend(loc='best', fontsize=12)
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Binned/smoothed attention weights
        for i in range(num_granularity_levels):
            label = f'Level {i}' + (' (Coarsest)' if i == 0 else ' (Finest)' if i == num_granularity_levels-1 else '')
            ax2.plot(
                bin_centers, 
                binned_weights[:, i], 
                marker='o', 
                markersize=8,
                linestyle='-', 
                linewidth=3,
                color=colors[i],
                label=label
            )
        
        ax2.set_xlabel('Normalized Timestep (1=earliest, 0=latest)', fontsize=14)
        ax2.set_ylabel('Average Attention Weight', fontsize=14)
        ax2.set_title('Router Preference Across Diffusion Process (Binned)', fontsize=16)
        ax2.legend(loc='best', fontsize=12)
        ax2.grid(True, alpha=0.3)
        
        # Add vertical guides for key timestep regions
        for x in [0.25, 0.5, 0.75]:
            ax1.axvline(x=x, color='gray', linestyle='--', alpha=0.4)
            ax2.axvline(x=x, color='gray', linestyle='--', alpha=0.4)
        
        # Annotate key regions
        ax2.text(0.9, 0.05, "Early\ndenoising", ha='center', fontsize=12)
        ax2.text(0.5, 0.05, "Middle\ndenoising", ha='center', fontsize=12)
        ax2.text(0.1, 0.05, "Late\ndenoising", ha='center', fontsize=12)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    save_path = 'Viz/subj01/time_router_analysis.png'
    unet = UNet2DConditionModel.from_pretrained('stabilityai/stable-diffusion-xl-base-1.0', subfolder='unet', cache_dir='data/cache')
    router.eval()
    total_steps = 1000

    timesteps = []
    attention_weights = []
    noisy_latents = torch.randn(1, 4, 28, 28).to(device)  # Example input
    
    with torch.no_grad():
        # Process all timesteps
        for t in tqdm(range(total_steps)):
            timestep = torch.tensor(t, device=device)

            # Normalize timestep (1=earliest, 0=latest)
            normalized_t = t / (total_steps - 1)
            
            # Create sinusoidal embeddings
            t_emb = unet.get_time_embed(sample=noisy_latents, timestep=timestep)
            
            # Forward pass through router
            attn_weights, _ = router(
                t_emb=t_emb,
                time_step=timestep,
                multiply_prior=True
            )
            
            # Store results
            timesteps.append(normalized_t)
            attention_weights.append(attn_weights.mean(0).cpu().numpy())

    # Convert to arrays
    timesteps = np.array(timesteps)
    attention_weights = np.array(attention_weights)
    
    # Get number of granularity levels
    num_granularity_levels = attention_weights.shape[1]
    
    # Bin the timesteps
    print("Binning and aggregating data...")
    
    bins = np.linspace(0, 1, num_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    # Initialize arrays for binning
    binned_weights = np.zeros((num_bins, num_granularity_levels))
    bin_counts = np.zeros(num_bins)
    
    # Assign data to bins
    for t, weights in zip(timesteps, attention_weights):
        bin_idx = np.digitize(t, bins) - 1
        if 0 <= bin_idx < num_bins:
            binned_weights[bin_idx] += weights
            bin_counts[bin_idx] += 1
    
    # Calculate averages
    for i in range(num_bins):
        if bin_counts[i] > 0:
            binned_weights[i] /= bin_counts[i]
    
    # Plot results
    print("Plotting results...")
    plot_router_preferences(
        timesteps, 
        attention_weights, 
        bin_centers, 
        binned_weights,
        num_granularity_levels,
        save_path
    )
    
    return {
        'timesteps': timesteps,
        'raw_attention_weights': attention_weights,
        'bin_centers': bin_centers,
        'binned_weights': binned_weights
    }


def add_recording_hooks(router):
    """Add hooks to record gate weights and granularity probs"""
    """Specifically for DiffuseSpaceRouterSelfAttn and DiffuseTimeRouterAttn"""
    hooks = []
    gate_weights_storage = {i: [] for i in range(router.num_granularity_levels)}
    granularity_probs_storage = []
    
    # Hooks for space routers
    for i, space_router in enumerate(router.space_routers):
        def get_hook(level_idx):
            def hook(module, input, output):
                gate_weights = output[1].detach().cpu()  # Assuming output[1] is gate weights
                gate_weights_storage[level_idx].append(gate_weights.detach().cpu())
            return hook
        
        handle = space_router.register_forward_hook(get_hook(i))
        hooks.append(handle)
    
    # Hook for time router if using soft routing
    if router.soft_time_routing and router.enable_time:
        def time_router_hook(module, input, output):
            # Assuming output[0] is granularity_probs
            granularity_probs = output[0].detach().cpu()
            granularity_probs_storage.append(granularity_probs)
        
        handle = router.time_router.register_forward_hook(time_router_hook)
        hooks.append(handle)
    
    return hooks, gate_weights_storage, granularity_probs_storage

def remove_hooks(hooks):
    """Remove all registered hooks"""
    for handle in hooks:
        handle.remove()

def get_time_space_weights(model, diffusion_prior, router, dataloader, device):
    output_dir = 'Viz/subj01/routers'

    unet = UNet2DConditionModel.from_pretrained('stabilityai/stable-diffusion-xl-base-1.0', subfolder='unet', cache_dir='data/cache')

    hooks, gate_weights_storage, granularity_probs_storage = add_recording_hooks(router)
    timesteps = np.linspace(0, 1000, 100).astype(int)
    noisy_latents = torch.randn(1, 4, 28, 28).to(device)  # Example input

    model.eval()
    router.eval()
    diffusion_prior.eval()

    all_gate_weights, all_granularity_probs = [], []

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            voxel_tensor = batch['voxel'].to(device).to(torch.float32)
            assert voxel_tensor.shape[1] == 3
            
            # Clear previous data
            for level_idx in gate_weights_storage:
                gate_weights_storage[level_idx].clear()
            granularity_probs_storage.clear()

            for j in range(3):
                vox = voxel_tensor[:, j]
                if vox.dim() == 2:  # Batch x Voxels
                    vox = vox.unsqueeze(1)  # Add channel dimension
                backbones, _, _, _ = model([vox], [1], training=False, return_exp_out=True)

                for t in tqdm(timesteps, desc="Processing timesteps"):
                    timestep = torch.tensor(t, device=device)
                    t_emb = unet.get_time_embed(sample=noisy_latents, timestep=timestep).expand(backbones[0][0].shape[0], -1)

                    route_out = router(t_emb, backbones, t, 1000, training=False)
                    # _, route_out = diffusion_prior(text_embed=route_out, image_embed=torch.zeros_like(route_out).to(route_out.device))

            # aggregate timesteps
            gate_weights = {k: torch.stack(v, dim=0) for k, v in gate_weights_storage.items()}  # T*3 E B L 1
            granularity_probs = torch.stack(granularity_probs_storage, dim=0)

            # averaging
            for level_idx in gate_weights:
                tmp = torch.split(gate_weights[level_idx], len(timesteps), dim=0)
                gate_weights[level_idx] = torch.stack(tmp, dim=0).mean(0)
            granularity_probs = torch.split(granularity_probs, len(timesteps), dim=0)
            granularity_probs = torch.stack(granularity_probs, dim=0).mean(0)
            
            all_gate_weights.append(gate_weights)
            all_granularity_probs.append(granularity_probs)

    all_gate_weights = {k: torch.cat([v[k] for v in all_gate_weights], dim=2) for k in all_gate_weights[0].keys()}
    all_granularity_probs = torch.cat(all_granularity_probs, dim=1)

    # Save the results
    all_gate_weights = {k: v.cpu().numpy() for k, v in all_gate_weights.items()}
    np.savez_compressed(os.path.join(output_dir, 'gate_weights.npz'), all_gate_weights)
    np.save(os.path.join(output_dir, 'granularity_probs.npy'), all_granularity_probs.numpy())
    
    remove_hooks(hooks)


def analyze_expert_within_level():
    """Analyze expert utilization within each granularity level"""
    output_dir = 'Viz/subj01/routers'
    gate_weights_data = np.load(os.path.join(output_dir, 'gate_weights.npz'), allow_pickle=True)['arr_0'][()]
    num_granularity_levels = len(gate_weights_data)

    results = {}
    
    for level_idx in range(num_granularity_levels):
        level_results = []
        
        # Gate weights for this level and timestep: (E, B, L, 1)
        gate_weights = gate_weights_data[level_idx].mean(0)

        avg_weights = gate_weights.mean((2, 3))

        # normalize the weights to 0-1
        # avg_weights = avg_weights - avg_weights.min(0, keepdims=True)
        # avg_weights = avg_weights / (avg_weights.max(0, keepdims=True) + 1e-8)
    
        results[level_idx] = avg_weights
    
    fig, axes = plt.subplots(num_granularity_levels, 1, 
                            figsize=(12, 4 * num_granularity_levels))
    
    if num_granularity_levels == 1:
        axes = [axes]
    
    for level_idx in range(num_granularity_levels):
        ax = axes[level_idx]
        box_data = [results[level_idx][e, :] for e in range(results[level_idx].shape[0])]
        sns.violinplot(data=box_data, ax=ax, inner='quartile', bw=0.2)
        ax.set_title(f'Expert Utilization Distribution (Level {level_idx})')
        ax.set_xlabel('Experts')
        ax.set_ylabel('Density')

    plt.tight_layout()
    plt.show()
    plt.savefig(os.path.join(output_dir, 'expert_utilization.png'), dpi=300, bbox_inches='tight')


def analyze_expert_across_level():
    """Analyze end-to-end expert contributions"""
    results = {}
    output_dir = 'Viz/subj01/routers'
    gate_weights_data = np.load(os.path.join(output_dir, 'gate_weights.npz'), allow_pickle=True)['arr_0'][()]
    granularity_probs_data = np.load(os.path.join(output_dir, 'granularity_probs.npy'))
    num_granularity_levels = len(gate_weights_data)
    num_experts_per_granularity= {0: 2, 1: 4, 2: 8, 3: 16}
    timesteps = np.linspace(0, 1000, 10).astype(int)
    
    for level_idx in range(num_granularity_levels):
        level_results = {}
        
        for expert_idx in range(num_experts_per_granularity[level_idx]):
            expert_results = []
            
            for t_idx, timestep in enumerate(timesteps):
                # Gate weights for this expert: (B, L, 1)
                gate_weights = gate_weights_data[level_idx][t_idx][expert_idx]
                
                # Granularity probs: (B, num_levels)
                gran_probs = granularity_probs_data[t_idx]
                
                # Level probs for each batch: (B, 1, 1)
                level_probs = gran_probs[:, level_idx][:, None, None]
                
                # Combined weight: (B, L, 1)
                combined_weights = gate_weights * level_probs
                
                # Average contribution
                avg_contribution = combined_weights.mean().item()
                expert_results.append((timestep, avg_contribution))
            
            level_results[expert_idx] = expert_results
        
        results[level_idx] = level_results
    
    latest_contributions = np.zeros((num_granularity_levels, max(num_experts_per_granularity.values())))
    
    for level_idx in range(num_granularity_levels):
        level_data = results.get(level_idx, {})
        for expert_idx in range(num_experts_per_granularity[level_idx]):
            expert_data = level_data.get(expert_idx, [])
            if expert_data:
                latest_contributions[level_idx, expert_idx] = expert_data[-1][1]
    
    # Create mask for unused experts
    mask = np.ones_like(latest_contributions, dtype=bool)
    for i in range(num_granularity_levels):
        mask[i, :num_experts_per_granularity[i]] = False
    
    # Plot heatmap
    plt.figure(figsize=(14, 8))
    sns.heatmap(latest_contributions, mask=mask, annot=True, fmt=".3f", cmap="viridis",
               xticklabels=[f"Expert {j}" for j in range(max(num_experts_per_granularity))],
               yticklabels=[f"Level {i}" for i in range(num_granularity_levels)])
    
    plt.title("End-to-End Expert Contributions")
    plt.tight_layout()
    heatmap_fig = plt.gcf()
    plt.show()
    
    # 2. Create time series plots for each level
    fig, axes = plt.subplots(num_granularity_levels, 1, 
                            figsize=(12, 4 * num_granularity_levels),
                            sharex=True)
    
    if num_granularity_levels == 1:
        axes = [axes]
    
    for level_idx in range(num_granularity_levels):
        ax = axes[level_idx]
        level_data = results.get(level_idx, {})
        
        for expert_idx in range(num_experts_per_granularity[level_idx]):
            expert_data = level_data.get(expert_idx, [])
            if expert_data:
                timesteps = [data[0] for data in expert_data]
                contributions = [data[1] for data in expert_data]
                ax.plot(timesteps, contributions, marker='o', label=f"Expert {expert_idx}")
        
        ax.set_title(f"Expert Contributions - Granularity Level {level_idx}")
        ax.set_ylabel("Average Contribution")
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True)
    
    axes[-1].set_xlabel("Diffusion Timestep")
    plt.tight_layout()
    plt.show()
    

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_name = 'single_subj1_40sess_ViT-BigG-img-prior-noblur-4096-BrainMoEMulti2-2L3F2--all+'
    lora_ckpt = 'MoEMulti2-noMeta-SpaceRouterSelfAttn-TimeRouterAttn_temp1.5_soft-routeBefore--all+'
    output_dir = 'Viz/subj01'

    model, diffusion_prior, diffuse_router, clip_embedder, test_dataloader, subj_list = load_model_and_data(model_name, lora_ckpt, load_image=True)
    model.to(device)
    model.eval()

    # routing_info = analyze_MoE_routing_patterns(model, test_dataloader, device=device)
    # attribute_ridge(model, test_dataloader, device=device, target_layer_idx=3)
    # analyze_time_router(diffuse_router.time_router, test_dataloader, device=device)
    
    # get_time_space_weights(model, diffusion_prior, diffuse_router, test_dataloader, device=device)
    # analyze_expert_within_level()
    analyze_expert_across_level()