import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import nibabel as nib
from nilearn.maskers import NiftiMasker
from nilearn import plotting
from nilearn.image import index_img, load_img, math_img, resample_to_img
from nilearn import datasets
from nilearn.input_data import NiftiLabelsMasker
from nilearn.regions import connected_regions
import os


n_images = 1000
category_names = ['accessory', 'animal', 'appliance', 'electronic', 'food', 'furniture', 'indoor', 'kitchen', 'outdoor', 'person', 'sports', 'vehicle']
n_categories = len(category_names)

label_matrix = np.load('data/NSD/coco_labels_val.npy')[:1000]

root = 'Viz/subj01/layerwise_mask'
num_experts_per_level = {0: 2, 1: 4, 2: 8, 3: 16}


def plot_per_expert_heatmaps(save_path=None, normalize_per_level=True):
    level_data = {}
    for level in range(3,4):
        n_experts = num_experts_per_level[level]
        
        # Collect data for all experts in this level
        level_attrs = np.zeros((n_experts, n_categories))
        expert_names = [f'Expert {i}' for i in range(n_experts)]
        # mask = np.load(f'Viz/subj01/ridge_grads_mask-layer{level}.npy')[:1000]
        # mask = np.transpose(mask, (1, 2, 3, 4, 0))
        
        for expert_id in range(n_experts):
            # Load the attribution data
            attribution_file = os.path.join(root, f'shap_attrs_level{level}_bproj_{expert_id}.nii.gz')
            attribution_data = nib.load(attribution_file).get_fdata()
            attribution_data = np.nan_to_num(attribution_data)  # Replace NaNs with 0
            # attribution_data = attribution_data * mask[expert_id]

            attribution_data = attribution_data.reshape(-1, n_images).mean(0)  # Average over voxels

            # Map attribution data to categories
            category_attrs = np.array([np.mean(attribution_data[label_matrix[:, i] == 1]) 
                                      for i in range(n_categories)])
            
            # Store in our level-specific array
            level_attrs[expert_id] = category_attrs
        
        # Scale for better readability (original values are very small)
        level_attrs = level_attrs * 1e6
        
        # Normalize within this level if requested
        if normalize_per_level:
            level_max = np.abs(level_attrs).max()
            if level_max > 0:  # Avoid division by zero
                level_attrs = level_attrs / level_max

        level_attrs = np.abs(level_attrs) 
        
        # Store for this level
        level_data[level] = {
            'attrs': level_attrs,
            'expert_names': expert_names
        }
        
        # Create dataframe for this level
        df = pd.DataFrame(level_attrs, columns=category_names, index=expert_names)
        
        # Plot each level separately
        plt.figure(figsize=(12, n_experts))
        
        # Determine colormap center for better visualization
        vmax = np.abs(df.values).max()
        vmin = 0
        
        # Create the heatmap
        ax = sns.heatmap(df, cmap='flare', annot=True, fmt=".2f", 
                     cbar_kws={'label': 'Attribution Value'}, vmin=vmin, vmax=vmax)
        
        plt.title(f'Level {level} - Expert Contribution per Category', fontsize=16)
        plt.xlabel('Categories', fontsize=12)
        plt.ylabel('Experts', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(os.path.join(save_path, f'level_{level}_experts_heatmap.png'), dpi=300, bbox_inches='tight')
        else:
            plt.show()


if __name__ == "__main__":
    plot_per_expert_heatmaps()

