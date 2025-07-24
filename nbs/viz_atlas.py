import os
import pandas as pd
import numpy as np
import seaborn as sns
from scipy import stats
from statsmodels.stats.multitest import multipletests
from sklearn.metrics import r2_score
from sklearn.decomposition import FastICA
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from nilearn.maskers import NiftiMasker
from nilearn import plotting
from nilearn.image import index_img, load_img, math_img, resample_to_img, threshold_img
from nilearn import datasets
from nilearn.input_data import NiftiLabelsMasker
from nilearn.regions import connected_regions
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import nibabel as nib

from viz_surface import normalize_nifti

def get_label_mapping(xml_path):
    import xml.etree.ElementTree as ET

    tree = ET.parse(xml_path)
    root = tree.getroot()

    label_map = {}
    for label_element in root.findall('./data/label'):
        try:
            index = int(label_element.get('index')) # Get the index attribute
            name = label_element.text.strip()      # Get the text content as the name
            if index is not None and name:
                label_map[index] = name
        except (AttributeError, ValueError, TypeError) as e:
            print(f"Warning: Could not parse label element: {label_element} - {e}")

    return label_map

def generate_distinct_colormap(n):
    if n < 10:
        base_cmap = plt.get_cmap('tab10')
    else:
        base_cmap = plt.get_cmap('tab20')
    colors = [base_cmap(i % 10) for i in range(n)]
    return ListedColormap(colors)

def analyze_attr(file, top_regions=5, atlas='visfAtlas', threshold=0.1):
    plt.style.use('default')
    data = nib.load(file)
    affine = data.affine
    data = normalize_nifti(data.get_fdata() * 10000, method='zscore')
    data = data / data.max()
    data = nib.Nifti1Image(data, affine=affine)
    data = threshold_img(data, threshold='99.5%')
    output_dir = '/'.join(file.split('/')[:-1])
    output_dir = os.path.join(output_dir, file.split('/')[-1].split('.')[0])
    os.makedirs(output_dir, exist_ok=True)

    atlas_img = None
    atlas_labels = None
    if atlas == 'Harvard-Oxford':
        atlas_data = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm')
        atlas_img = atlas_data.maps
        atlas_labels = atlas_data.labels
        atlas_labels = atlas_labels[1:]  # Remove background label
    elif atlas == 'visfAtlas':
        ref = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm')
        atlas_img = nib.load('Viz/visfAtlas/nifti_volume/visfAtlas_MNI152_volume.nii.gz')
        atlas_img = resample_to_img(atlas_img, ref.maps, interpolation='nearest')  # resample to 2mm
        atlas_data = atlas_img.get_fdata()

        atlas_labels = np.unique(atlas_data)
        label_map = get_label_mapping('Viz/visfAtlas/nifti_volume/visfAtlas_FSL.xml')
        atlas_labels = [label_map[label] for label in atlas_labels if label in label_map]
    elif atlas == 'schaefer_2018':
        atlas_data = datasets.fetch_atlas_schaefer_2018(n_rois=1000, yeo_networks=17, resolution_mm=2)
        atlas_img = nib.load(atlas_data.maps)
        atlas_labels = atlas_data.labels.tolist()
        atlas_labels = [x.decode('utf-8') for x in atlas_labels]

    coords = plotting.find_xyz_cut_coords(data)

    fig = plt.figure(figsize=(15, 5))
    display = plotting.plot_stat_map(data,
                        display_mode='ortho', # Show axial, sagittal, coronal
                        cut_coords=coords, # Center plot roughly on activation
                        threshold=threshold, # Example threshold: show values > 2 (adjust based on component values)
                        draw_cross=False,
                        figure=fig)
    fig.savefig(os.path.join(output_dir, f'overlay.png'), dpi=300)
    plt.show()
    plt.close(fig)

    fig = plt.figure(figsize=(15, 5))
    display = plotting.plot_stat_map(data,
                        display_mode='mosaic',
                        threshold=threshold, # Example threshold: show values > 2 (adjust based on component values)
                        figure=fig)
    fig.savefig(os.path.join(output_dir, f'overlay_mosic.png'), dpi=300)
    plt.show()
    plt.close(fig)

    # glass brain
    fig = plt.figure(figsize=(15, 5))
    plotting.plot_glass_brain(
            data,
            # threshold=threshold,
            colorbar=True,
            figure=fig
        )
    plt.show()
    fig.savefig(os.path.join(output_dir, f'glass_brain.png'), dpi=300)
    plt.close(fig)

    if not np.allclose(data.affine, atlas_img.affine):
        print("WARNING: Affine matrices of components and atlas do not match!")
        print("Components Affine:\n", data.affine)
        print("Atlas Affine:\n", atlas_img.affine)
        resampling_setting = 'labels' # Resample data ('components_img') to atlas space
    else:
        print("Affine matrices match. No resampling needed by masker based on affine.")
        resampling_setting = None # No resampling needed if affines match

    masker = NiftiLabelsMasker(
        labels_img=atlas_img,
        labels=atlas_labels,
        strategy='mean', # Calculate the mean value within each label ROI
        standardize=False, # VERY IMPORTANT: Do not standardize ICA component values
        resampling_target=resampling_setting, # 'labels', 'data', or None
        # Ensures data is resampled to atlas space if needed
        verbose=1 # Show progress
    )
    region_signals = masker.fit_transform(data)
    results_df = pd.DataFrame(region_signals, columns=atlas_labels)

    print(results_df)
    os.makedirs(os.path.join(output_dir, atlas), exist_ok=True)
    results_df.to_csv(os.path.join(output_dir, atlas, f'atlas_results.csv'))

    print('------Top 5 dominant regions------')
    # For each row in results_df, find the top 5 dominant regions
    for idx, row in results_df.iterrows():
        # Sort by absolute value to find the strongest signals (positive or negative)
        top_regions = row.abs().sort_values(ascending=False)[:top_regions]
        print(f"{idx}:")
        for region, value in top_regions.items():
            # Use the original value (not absolute) for display
            original_value = row[region]
            print(f"  {region}: {original_value:.4f}")

        # Visualize each top region in the brain
        top_regions = list(top_regions.index)
        
        # Find this region in the atlas
        if atlas == 'visfAtlas':
            # For visfAtlas, find the numeric label for this region name
            label_map_inv = {v: k for k, v in label_map.items()}
            region_values = [label_map_inv[r] for r in top_regions]
        else:
            # For other atlases, find the index in atlas_labels
            region_values = [atlas_labels.index(r) for r in top_regions]

        region_mask = np.zeros(atlas_img.shape, dtype=int)
        for i, v in enumerate(region_values):
            region_mask[atlas_img.get_fdata() == (v + 1)] = i + 1
        region_mask = nib.Nifti1Image(region_mask, atlas_img.affine, atlas_img.header)
        top_regions = [r.replace('17Networks_', '') for r in top_regions]

        with open(os.path.join(output_dir, atlas, f'top_regions.txt'), 'w') as f:
            f.write(' | '.join(top_regions))
        
        # Plot this region
        fig = plt.figure(figsize=(15, 5))
        display = plotting.plot_roi(
            region_mask,
            # title=f"Regions: {top_regions}",
            display_mode='ortho',
            # cut_coords=plotting.find_xyz_cut_coords(region_mask),
            figure=fig,
            draw_cross=False,
            cmap=generate_distinct_colormap(len(top_regions)),
            colorbar=True,
        )
        fig.suptitle('|'.join(top_regions), fontsize=20)
        os.makedirs(os.path.join(output_dir, atlas), exist_ok=True)
        plt.savefig(os.path.join(output_dir, atlas, f'atlas.png'), dpi=300)
        plt.show()
        plt.close(fig)

        fig = plt.figure(figsize=(15, 5))
        display = plotting.plot_roi(
            region_mask,
            display_mode='mosaic',
            figure=fig,
            draw_cross=False,
            cmap=generate_distinct_colormap(len(top_regions)),
            colorbar=True,
        )
        os.makedirs(os.path.join(output_dir, atlas), exist_ok=True)
        plt.savefig(os.path.join(output_dir, atlas, f'atlas_mosaic.png'), dpi=300)
        plt.show()
        plt.close(fig)

    return results_df

if __name__ == "__main__":
    file = os.listdir('Viz/subj01/layerwise_mask/masked-average-mni')
    tops = {0: 8, 1: 4, 2: 2, 3: 1}

    for f in file:
        if 'nii.gz' not in f: continue
        
        if 'level0' in f:
            level_idx = 0
        elif 'level1' in f:
            level_idx = 1
        elif 'level2' in f:
            level_idx = 2
        elif 'level3' in f:
            level_idx = 3
        top = tops[level_idx]

        analyze_attr(os.path.join('Viz/subj01/layerwise_mask/masked-average-mni', f), atlas='schaefer_2018', threshold=0, top_regions=top)