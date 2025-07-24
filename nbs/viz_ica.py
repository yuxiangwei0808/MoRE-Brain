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
import umap
from nilearn.maskers import NiftiMasker
from nilearn import plotting
from nilearn.image import index_img, load_img, math_img, resample_to_img
from nilearn import datasets
from nilearn.input_data import NiftiLabelsMasker
from nilearn.regions import connected_regions
import matplotlib.pyplot as plt
import nibabel as nib

from viz_atlas import generate_distinct_colormap

def norm_by_subject(data):
    mask = np.load('data/NSD/ROI/mask_union_mni.npy')
    n_images_per_subject = [1000, 1000, 930, 907, 1000, 930, 1000, 907]
    start = 0
    for i in range(len(n_images_per_subject)):
        end = start + n_images_per_subject[i]
        # data[..., start:end] = (data[..., start:end] - np.mean(data[..., start:end])) / np.std(data[..., start:end])
        # std_value = data[mask, start:end][1200:2000].std()
        data[..., start:end] /= (np.std(data[..., start:end], axis=1, keepdims=True) + 1e-6)
        start = end
    return data

def get_ica(n_components, norm=True):
    path = 'Viz/all_shap_attrs_mni.nii.gz'  # Path to the subject images
    subject_images = nib.load(path)
    affine = subject_images.affine
    subject_images = subject_images.get_fdata()
    masking = False
    output_dir = f'Viz/ica/ica_output_{n_components}_stdDiv_perSubj'

    subject_ids = [f'Sub{i+1}' for i in range(8)]
    group_labels = ['subj01', 'subj02', 'subj03', 'subj04', 'subj05', 'subj06', 'subj07', 'subj08']

    if norm:
        assert not masking
        subject_images = norm_by_subject(subject_images)
        print('Normalized by subject')

    if masking:
        # Use NiftiMasker to handle masking and convert to a 2D numpy array (subjects x voxels)
        masker = NiftiMasker(mask_img='Viz/visual_mask_MNI', standardize=False) # Don't standardize subject means
        voxel_by_subject_data = masker.fit_transform(subject_images).T   # vox x subjects
    else:
        img_shape = subject_images.shape
        voxel_by_subject_data = np.reshape(subject_images, (-1, subject_images.shape[-1]))

    print('Begin')
    ica = FastICA(n_components=n_components, random_state=0, whiten='unit-variance', max_iter=1000)
    subject_loadings = ica.fit_transform(voxel_by_subject_data.T) # Fit on (Subjects x Voxels) -> returns (Subjects x Components)
    # Get the spatial maps (Components x Voxels)
    spatial_maps_voxels = ica.components_

    if masking:
        spatial_maps_nifti = masker.inverse_transform(spatial_maps_voxels)
    else:
        spatial_maps_voxels = np.reshape(spatial_maps_voxels.T, img_shape[:-1] + (n_components,))
        spatial_maps_nifti = nib.Nifti1Image(spatial_maps_voxels, affine=affine)

    spatial_maps_nifti.to_filename(f'{output_dir}.nii.gz')
    np.save(f'{output_dir}_loadings.npy', subject_loadings)

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

def plot_components(ica_file, component_idx, atlas='schaefer_2018_1000_17'):
    plt.style.use('default')
    components_img = nib.load(ica_file)
    out_fn = ica_file.split('/')[-1].replace('.nii.gz', '')
    output_dir = f'Viz/ica/per_components/{out_fn}'
    os.makedirs(output_dir, exist_ok=True)

    bg_img = datasets.load_mni152_template(resolution=2)
    component_img = index_img(components_img, component_idx) # Extract the i-th component
    affine = component_img.affine

    component_img = component_img.get_fdata()
    for i in range(component_img.shape[-1]):
        component_img[..., i] = component_img[..., i] / np.max(np.abs(component_img[..., i]))
    component_img = np.max(component_img, axis=-1)
    component_img = nib.Nifti1Image(component_img, affine=affine)

    coords = plotting.find_xyz_cut_coords(component_img)

    # Plot slices
    fig = plt.figure(figsize=(15, 5))
    display = plotting.plot_stat_map(component_img,
                        display_mode='ortho', # Show axial, sagittal, coronal
                        cut_coords=coords, # Center plot roughly on activation
                        threshold=0.01, # Example threshold: show values > 2 (adjust based on component values)
                        figure=fig)
    plt.show()
    # fig.savefig(os.path.join(output_dir, f'component_{component_idx+1}_overlay.png'), dpi=300)
    plt.close(fig)

    if atlas == 'schaefer_2018_1000_17':
        atlas_data = datasets.fetch_atlas_schaefer_2018(n_rois=1000, yeo_networks=17, resolution_mm=2)
        atlas_img = nib.load(atlas_data.maps)
        atlas_labels = atlas_data.labels.tolist()
        atlas_labels = [x.decode('utf-8') for x in atlas_labels]
    
    all_atlas_regions = []
    if not np.allclose(components_img.affine, atlas_img.affine):
        print("WARNING: Affine matrices of components and atlas do not match!")
        print("Components Affine:\n", components_img.affine)
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
        verbose=0 # Show progress
    )

    all_top_region_names = []
    for idx in component_idx:
        component_img = index_img(components_img, idx) # Extract the i-th component
        region_signals = masker.fit_transform(component_img)
        component_names = [f'Component {idx}']
        region_signals_df = pd.DataFrame(region_signals, index=component_names, columns=atlas_labels)
        top_regions = region_signals_df.abs().apply(lambda x: x.nlargest(5).index.tolist(), axis=1)
        top_regions_indices = [atlas_labels.index(region) for region in top_regions[0]] 

        top_regions = [region.replace('17Networks_', '') for region in top_regions[0]]
        all_top_region_names.extend(top_regions)
        title = '|'.join(top_regions)

        region_mask = np.zeros(atlas_img.shape, dtype=int)
        for i, v in enumerate(top_regions_indices):
            region_mask[atlas_img.get_fdata() == (v + 1)] = i + 1
        region_mask = nib.Nifti1Image(region_mask, atlas_img.affine, atlas_img.header)

        fig = plt.figure(figsize=(15, 5))
        display = plotting.plot_roi(
            region_mask,
            display_mode='ortho',
            figure=fig,
            draw_cross=False,
            cmap='CMRmap_r',
            colorbar=False,
            cut_coords=plotting.find_xyz_cut_coords(region_mask),
        )
        fig.suptitle(title, fontsize=20)
        # plt.savefig(os.path.join(output_dir, f'all_top_regions.png'), dpi=300)
        plt.show()
        plt.close(fig)
    print('Top regions:', all_top_region_names)
    
def analyze_ica(ica_file, template='MNI152_T1_2mm', atlas='Harvard-Oxford', n_components=3, threshold=0.01, top_k=3):
    plt.style.use('default')
    components_img = nib.load(ica_file)
    out_fn = ica_file.split('/')[-1].replace('.nii.gz', '')
    output_dir = f'Viz/ica/{atlas}/components/{out_fn}'
    os.makedirs(output_dir, exist_ok=True)

    if template == 'MNI152_T1_2mm':
        bg_img = datasets.load_mni152_template(resolution=2)
    elif template == 'MNI152_T1_1mm':
        bg_img = datasets.load_mni152_template(resolution=1)
    else:
        bg_img = None  # Let nilearn choose default

    # Load atlas for region identification
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
    elif atlas == 'schaefer_2018_1000_17':
        atlas_data = datasets.fetch_atlas_schaefer_2018(n_rois=1000, yeo_networks=17, resolution_mm=2)
        atlas_img = nib.load(atlas_data.maps)
        atlas_labels = atlas_data.labels.tolist()
        atlas_labels = [x.decode('utf-8') for x in atlas_labels]

    for i in range(min(n_components, 8)):
        component_img = index_img(components_img, i) # Extract the i-th component
        title = f'Independent Component {i+1}'
        
        coords = plotting.find_xyz_cut_coords(component_img)

        # Plot slices
        fig = plt.figure(figsize=(15, 5))
        display = plotting.plot_stat_map(component_img, bg_img=bg_img,
                            title=title, display_mode='ortho', # Show axial, sagittal, coronal
                            cut_coords=coords, # Center plot roughly on activation
                            threshold=threshold, # Example threshold: show values > 2 (adjust based on component values)
                            figure=fig)
        fig.savefig(os.path.join(output_dir, f'component_{i+1}_overlay.png'), dpi=300)
        plt.close(fig)

        # 2. Create a glass brain visualization
        fig = plt.figure(figsize=(10, 5))
        plotting.plot_glass_brain(
            component_img,
            title=f'{title} - Glass Brain',
            threshold=threshold,
            colorbar=True,
            figure=fig
        )
        fig.savefig(os.path.join(output_dir, f'component_{i+1}_glass.png'), dpi=300)
        plt.close(fig)

    # plot all components
    all_comp = components_img.get_fdata()
    for i in range(all_comp.shape[-1]):
        all_comp[..., i] = all_comp[..., i] / np.max(np.abs(all_comp[..., i]))
    all_comp = np.max(all_comp, axis=-1)
    all_comp = nib.Nifti1Image(all_comp, affine=components_img.affine)

    fig = plt.figure(figsize=(15, 5))
    coords = plotting.find_xyz_cut_coords(all_comp)
    display = plotting.plot_stat_map(all_comp, 
                        display_mode='ortho', # Show axial, sagittal, coronal
                        cut_coords=coords, # Center plot roughly on activation
                        threshold=threshold, # Example threshold: show values > 2 (adjust based on component values)
                        figure=fig)
    plt.show()
    fig.savefig(os.path.join(output_dir, f'all_components_overlay.png'), dpi=300)
    plt.close(fig)

    fig = plt.figure(figsize=(15, 5))
    display = plotting.plot_stat_map(all_comp, bg_img=bg_img,
                        display_mode='mosaic', # Show axial, sagittal, coronal
                        threshold=threshold, # Example threshold: show values > 2 (adjust based on component values)
                        figure=fig)
    plt.show()
    fig.savefig(os.path.join(output_dir, f'all_components_overlay_mosaic.png'), dpi=300)
    plt.close(fig)    


    # 3. If atlas is provided, identify regions that overlap with the component
    if atlas_img is not None and atlas_labels is not None:
        print("\nChecking spatial alignment (affine matrix)...")
        if not np.allclose(components_img.affine, atlas_img.affine):
            print("WARNING: Affine matrices of components and atlas do not match!")
            print("Components Affine:\n", components_img.affine)
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

        region_signals = masker.fit_transform(components_img)
        component_names = [f'Component {i+1}' for i in range(n_components)]
        results_df = pd.DataFrame(region_signals, index=component_names, columns=atlas_labels)

        print("\n--- Average Component Values per Atlas Region ---")
        print(results_df)
        results_df.to_csv(os.path.join(output_dir, f'atlas_results.csv'))

        print("\n--- Top 3 Dominant Region per Component (Max Absolute Average Value) ---")
        # get top 3
        top_regions = results_df.abs().apply(lambda x: x.nlargest(3).index.tolist(), axis=1)
        top_values = results_df.abs().apply(lambda x: x.nlargest(3).values.tolist(), axis=1)
        top_signed_values = [results_df.loc[comp, regions] for comp, regions in top_regions.items()]
        top_summary_df = pd.DataFrame({
            f'Top {top_k} Regions': top_regions,
            f'Max Absolute Mean Values': top_values,
            f'Signed Mean Values in Top {top_k} Regions': top_signed_values
        })
        print(top_summary_df)
        top_summary_df.to_csv(os.path.join(output_dir, f'atlas_top{top_k}.csv'))

        # plot the dominant regions with the atlas
        for i in range(min(n_components, 8)):
            component_img = index_img(components_img, i) # Extract the i-th component
            component_name = component_names[i]
            print(f"\nVisualizing {component_name}...")
            dominant_region_names = top_regions[component_name]
            dominant_region_indices = [atlas_labels.index(region) for region in dominant_region_names]  # we need to consider background
            
            formula = ' | '.join([f'(img == {val + 1})' for val in dominant_region_indices])
            region_mask_img = math_img(formula, img=atlas_img)
            
            fig = plt.figure(figsize=(12, 5))
            plotting.plot_roi(
                roi_img=region_mask_img,
                title=f"{' | '.join(dominant_region_names)}",
                cut_coords=plotting.find_xyz_cut_coords(region_mask_img), # Use component peak coords for comparison
                display_mode='ortho',
                # bg_img=component_img,  # overlay with component
                # view_type='contours', # Example: draw contours instead of filled ROI
                # alpha=0.7, # Transparency if overlaying
            )
            plt.savefig(os.path.join(output_dir, f'component_{i+1}_dominant_regions.png'), dpi=300)
            plt.close()
        
        # plot all top regions across components
        all_region_values = results_df.abs().sum(axis=0)
        if atlas == 'schaefer_2018_1000_17':
            all_top_regions = all_region_values.nlargest(100).index.tolist()
        else:
            all_top_regions = all_region_values.nlargest(20).index.tolist()

        all_top_regions = list(set(all_top_regions))
        all_top_regions_values = [atlas_labels.index(region) for region in all_top_regions]
        
        region_mask = np.zeros(atlas_img.shape, dtype=int)
        for i, v in enumerate(all_top_regions_values):
            region_mask[atlas_img.get_fdata() == (v + 1)] = i + 1
        region_mask = nib.Nifti1Image(region_mask, atlas_img.affine, atlas_img.header)

        fig = plt.figure(figsize=(15, 5))
        display = plotting.plot_roi(
            region_mask,
            display_mode='ortho',
            figure=fig,
            draw_cross=False,
            cmap='CMRmap_r',
            colorbar=True,
        )
        plt.savefig(os.path.join(output_dir, f'all_top_regions.png'), dpi=300)
        plt.show()
        plt.close(fig)

        fig = plt.figure(figsize=(15, 5))
        display = plotting.plot_roi(
            region_mask,
            display_mode='mosaic',
            figure=fig,
            draw_cross=False,
            cmap='CMRmap_r',
            colorbar=True,
        )
        plt.show()
        plt.savefig(os.path.join(output_dir, f'all_top_regions_mosaic.png'), dpi=300)
        plt.close(fig)

        # plot 25 regions based on overall top regions
        for i, v in enumerate(all_top_regions_values[:25]):
            region = all_top_regions[i].replace('17Networks_', '')
            region_mask = np.zeros(atlas_img.shape, dtype=int)
            region_mask[atlas_img.get_fdata() == (v + 1)] = i + 1
            region_mask = nib.Nifti1Image(region_mask, atlas_img.affine, atlas_img.header)

            fig = plt.figure(figsize=(15, 5))
            display = plotting.plot_roi(
                region_mask,
                display_mode='ortho',
                figure=fig,
                draw_cross=True,
                cmap='autumn',
                colorbar=False,
            )
            fig.suptitle(region, fontsize=24)
            plt.show()
            os.makedirs(os.path.join(output_dir, 'overall_top_regions'), exist_ok=True)
            plt.savefig(os.path.join(output_dir, 'overall_top_regions', region), dpi=300)
            plt.close()

def analyze_ica_mixing(mixing_file, n_components):
    output_dir = mixing_file.split('/')[-1].split('_loadings.npy')[0]
    output_dir = f'Viz/ica/mixing_matrix/{output_dir}/subject_corr/'
    os.makedirs(output_dir, exist_ok=True)

    n_subjects = 8
    n_images_per_subject = [1000, 1000, 930, 907, 1000, 930, 1000, 907]

    mixing_matrix = np.load(mixing_file)

    ### visualize heatmap
    # Calculate average mixing coefficients for each subject
    avg_mixing = np.zeros((n_subjects, n_components))
    start_idx = 0
    for i in range(n_subjects):
        end_idx = start_idx + n_images_per_subject[i]
        # avg_mixing[i] = np.mean(np.abs(mixing_matrix[start_idx:end_idx]), axis=0)
        avg_mixing[i] = np.mean(mixing_matrix[start_idx:end_idx], axis=0)
        start_idx = end_idx

    plt.figure(figsize=(10, 8))
    sns.heatmap(avg_mixing, cmap="coolwarm", 
                xticklabels=[f"IC{i+1}" for i in range(n_components)],
                yticklabels=[f"Subject {i+1}" for i in range(n_subjects)],)
    plt.title("Average Mixing Coefficients per Subject")
    plt.xlabel("Independent Components")
    plt.ylabel("Subjects")
    plt.tight_layout()
    plt.show()
    plt.savefig(os.path.join(output_dir, 'avg_mixing_heatmap.png'), dpi=300)
    plt.close()

    ### Analyze how each component varies across subjects using boxplots and statistical tests
    component_data = []
    subject_labels = []
    component_labels = []
    
    for comp_idx in range(min(n_components, 6)):  # Limit to 6 components for clarity
        start_idx = 0
        for subj_idx in range(n_subjects):
            end_idx = start_idx + n_images_per_subject[subj_idx]
            # Get mixing coefficients for this component and subject
            coeffs = mixing_matrix[start_idx:end_idx, comp_idx]
            start_idx = end_idx

            component_data.extend(coeffs)
            subject_labels.extend([f"Subject {subj_idx+1}"] * len(coeffs))
            component_labels.extend([f"IC{comp_idx+1}"] * len(coeffs))
    
    # Create a DataFrame for easier plotting
    df = pd.DataFrame({
        "Component": component_labels,
        "Subject": subject_labels,
        "Coefficient": component_data
    })
    
    # Plot boxplots for each component across subjects
    plt.figure(figsize=(15, 8))
    for i, comp in enumerate(range(min(6, n_components))):
        plt.subplot(2, 3, i+1)
        comp_name = f"IC{comp+1}"
        sns.boxplot(x="Subject", y="Coefficient", 
                    data=df[df["Component"] == comp_name])
        plt.title(f"Component {comp+1} Variation")
        plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.show()
    plt.savefig(os.path.join(output_dir, 'component_variation_boxplots.png'), dpi=300)
    plt.close()
    
    # ANOVA to test for significant differences between subjects
    print("ANOVA Results: Component Variations Across Subjects")
    print("=" * 60)
    with open(os.path.join(output_dir, 'anova_results.txt'), 'w') as f:
        f.write("ANOVA Results: Component Variations Across Subjects\n")
        f.write("=" * 60 + "\n")
    
    for comp_idx in range(n_components):
        comp_data = []
        start_idx = 0
        for subj_idx in range(n_subjects):
            end_idx = start_idx + n_images_per_subject[subj_idx]
            comp_data.append(mixing_matrix[start_idx:end_idx, comp_idx])
            start_idx = end_idx
        
        # Perform one-way ANOVA
        f_val, p_val = stats.f_oneway(*comp_data)
        
        print(f"Component {comp_idx+1}: F={f_val:.3f}, p={p_val:.6f}", 
              "*" if p_val < 0.05 else "")
        with open(os.path.join(output_dir, 'anova_results.txt'), 'a') as f:
            f.write(f"Component {comp_idx+1}: F={f_val:.3f}, p={p_val:.6f}" +
                    (" *" if p_val < 0.05 else "") + "\n")
    
    print("=" * 60)
    print("* indicates significant differences between subjects (p<0.05)")
    with open(os.path.join(output_dir, 'anova_results.txt'), 'a') as f:
        f.write("=" * 60 + "\n")
        f.write("* indicates significant differences between subjects (p<0.05)\n")

    ### Test which subject best explains each component based on variance explained.
    variance_explained = np.zeros((n_subjects, n_components))
    
    for comp_idx in range(n_components):
        comp_coeffs = mixing_matrix[:, comp_idx]
        comp_var = np.var(comp_coeffs)
        
        start_idx = 0
        for subj_idx in range(n_subjects):
            end_idx = start_idx + n_images_per_subject[subj_idx]
            # Calculate proportion of total variance explained by this subject
            subj_coeffs = comp_coeffs[start_idx:end_idx]
            start_idx = end_idx
            
            subj_var = np.var(subj_coeffs)
            subj_mean = np.mean(subj_coeffs)
            overall_mean = np.mean(comp_coeffs)
            
            # Weighted contribution to variance
            variance_explained[subj_idx, comp_idx] = (n_images_per_subject[subj_idx] / len(comp_coeffs)) * (
                (subj_var + (subj_mean - overall_mean)**2) / comp_var
            )
    
    # Plot the results
    plt.figure(figsize=(10, 8))
    sns.heatmap(variance_explained, cmap="YlGnBu", 
                xticklabels=[f"IC{i+1}" for i in range(n_components)],
                yticklabels=[f"Subject {i+1}" for i in range(n_subjects)], 
                fmt=".2f")
    plt.title("Proportion of Variance Explained by Each Subject for Each Component")
    plt.xlabel("Independent Components")
    plt.ylabel("Subjects")
    plt.tight_layout()
    plt.show()
    plt.savefig(os.path.join(output_dir, 'variance_explained_heatmap.png'), dpi=300)
    plt.close()
    
    # Identify which subject best explains each component
    best_subjects = np.argmax(variance_explained, axis=0)
    print("Best Subject for Each Component:")
    print("=" * 50)
    with open(os.path.join(output_dir, 'best_subjects.txt'), 'w') as f:
        f.write("Best Subject for Each Component:\n")
        f.write("=" * 50 + "\n")

    for comp_idx in range(n_components):
        best_subj = best_subjects[comp_idx]
        print(f"Component {comp_idx+1}: Subject {best_subj+1} " +
              f"(Explains {variance_explained[best_subj, comp_idx]:.2%} of variance)")
        with open(os.path.join(output_dir, 'best_subjects.txt'), 'a') as f:
            f.write(f"Component {comp_idx+1}: Subject {best_subj+1} " +
                    f"(Explains {variance_explained[best_subj, comp_idx]:.2%} of variance)\n")
    print("=" * 50)
    with open(os.path.join(output_dir, 'best_subjects.txt'), 'a') as f:
        f.write("=" * 50 + "\n")
        f.write("Note: The variance explained is relative to the total variance of the component.\n")
    

def analyze_img_category(mixing_file, n_components):
    output_dir = mixing_file.split('/')[-1].split('_loadings.npy')[0]
    output_dir = f'Viz/ica/category/{output_dir}/single_category/'
    os.makedirs(output_dir, exist_ok=True)

    n_subjects = 8
    n_images_per_subject = [1000, 1000, 930, 907, 1000, 930, 1000, 907]
    category_names = ['accessory', 'animal', 'appliance', 'electronic', 'food', 'furniture', 'indoor', 'kitchen', 'outdoor', 'person', 'sports', 'vehicle']
    n_categories = len(category_names)

    label_matrix = np.load('data/NSD/coco_labels_val.npy')
    A = np.load(mixing_file)

    """How component relate to categorites (multi-label)"""
    n_top_components = 3
    comp_df = pd.DataFrame(A, columns=[f'IC{i+1}' for i in range(n_components)])
    
    # Calculate mean component values for images with each category
    category_component_means = np.zeros((n_categories, n_components))
    category_component_effect = np.zeros((n_categories, n_components))
    p_values = np.zeros((n_categories, n_components))
    
    for cat_idx, category in enumerate(category_names):
        # Images with this category
        with_category = label_matrix[:, cat_idx] == 1
        
        if np.sum(with_category) > 0 and np.sum(~with_category) > 0:
            for comp_idx in range(n_components):
                # Calculate mean for images with and without this category
                mean_with = np.mean(A[with_category, comp_idx])
                mean_without = np.mean(A[~with_category, comp_idx])
                
                # Store the mean for images with this category
                category_component_means[cat_idx, comp_idx] = mean_with
                
                # Calculate the effect size (difference from images without the category)
                category_component_effect[cat_idx, comp_idx] = mean_with - mean_without
                
                # Perform t-test
                t_stat, p_val = stats.ttest_ind(
                    A[with_category, comp_idx],
                    A[~with_category, comp_idx],
                    equal_var=False  # Welch's t-test
                )
                p_values[cat_idx, comp_idx] = p_val
    
    # Create DataFrames for results
    means_df = pd.DataFrame(
        category_component_means,
        index=category_names,
        columns=[f'IC{i+1}' for i in range(n_components)]
    )
    
    effect_df = pd.DataFrame(
        category_component_effect,
        index=category_names,
        columns=[f'IC{i+1}' for i in range(n_components)]
    )

    # Save the means and effects to CSV
    means_df.to_csv(os.path.join(output_dir, 'category_means.csv'))
    effect_df.to_csv(os.path.join(output_dir, 'category_effects.csv'))
    
    # Apply Benjamini-Hochberg FDR correction for multiple comparisons
    p_values_flat = p_values.flatten()
    reject, p_corrected, _, _ = multipletests(p_values_flat, method='fdr_bh')
    p_corrected = p_corrected.reshape(p_values.shape)
    
    significance_df = pd.DataFrame(
        p_corrected < 0.05,  # Significant after correction
        index=category_names,
        columns=[f'IC{i+1}' for i in range(n_components)]
    )
    
    # Visualize the effect size with significant relationships highlighted
    plt.figure(figsize=(14, 10))
    
    # Create a mask for non-significant values
    mask = ~significance_df.values
    
    # Plot heatmap
    sns.heatmap(effect_df, cmap="RdBu_r", center=0, 
                mask=mask, cbar_kws={'label': 'Component Effect (Presence - Absence)'})
    plt.title('Component Effect by Category (Only Significant Relationships Shown)')
    plt.tight_layout()
    plt.show()

    plt.savefig(os.path.join(output_dir, 'category_effects_heatmap.png'), dpi=300)
    plt.close()
    significance_df.to_csv(os.path.join(output_dir, 'category_significance.csv'))
    
    # Find top components for each category
    print(f"Top {n_top_components} Components for Each Category (Effect Size):")
    print("="*70)

    with open(os.path.join(output_dir, 'top_components.txt'), 'w') as f:
        f.write(f"Top {n_top_components} Components for Each Category (Effect Size):\n")
        f.write("="*70 + "\n")
    
    for category in category_names:
        # Get the absolute effect size
        abs_effect = effect_df.loc[category].abs()
        top_comps = abs_effect.nlargest(n_top_components).index.tolist()
        
        # Get the actual effect (with sign)
        effects = [f"{effect_df.loc[category, comp]:.3f}" for comp in top_comps]
        
        # Check significance
        sig_status = [
            "sig" if significance_df.loc[category, comp] else "ns" 
            for comp in top_comps
        ]
        
        print(f"{category}: {list(zip(top_comps, effects, sig_status))}")
        with open(os.path.join(output_dir, 'top_components.txt'), 'a') as f:
            f.write(f"{category}: {list(zip(top_comps, effects, sig_status))}\n")
    
    # Find top categories for each component
    print("\nTop Categories for Each Component:")
    print("="*70)
    with open(os.path.join(output_dir, 'top_categories.txt'), 'w') as f:
        f.write("\nTop Categories for Each Component:\n")
        f.write("="*70 + "\n")
    
    for comp in effect_df.columns:
        # Get the absolute effect size
        abs_effect = effect_df[comp].abs()
        top_cats = abs_effect.nlargest(3).index.tolist()
        
        # Get the actual effect (with sign)
        effects = [f"{effect_df.loc[cat, comp]:.3f}" for cat in top_cats]
        
        # Check significance
        sig_status = [
            "sig" if significance_df.loc[cat, comp] else "ns" 
            for cat in top_cats
        ]
        
        print(f"{comp}: {list(zip(top_cats, effects, sig_status))}")
        with open(os.path.join(output_dir, 'top_categories.txt'), 'a') as f:
            f.write(f"{comp}: {list(zip(top_cats, effects, sig_status))}\n")


def analyze_subject_category(mixing_file, n_components):
    output_dir = mixing_file.split('/')[-1].split('_loadings.npy')[0]
    output_dir = f'Viz/ica/category/{output_dir}/single_category_subj/'
    os.makedirs(output_dir, exist_ok=True)

    n_subjects = 8
    n_images_per_subject = [1000, 1000, 930, 907, 1000, 930, 1000, 907]
    category_names = ['accessory', 'animal', 'appliance', 'electronic', 'food', 'furniture', 'indoor', 'kitchen', 'outdoor', 'person', 'sports', 'vehicle']
    n_categories = len(category_names)

    label_matrix = np.load('data/NSD/coco_labels_val.npy')
    A = np.load(mixing_file)

    category_subj_means = np.zeros((n_categories * n_subjects, n_components))
    # category_subj_effect = np.zeros((n_categories, n_subjects))
    # p_values = np.zeros((n_categories , n_subjects))

    start_idx = 0
    for i in range(n_subjects):
        end_idx = start_idx + n_images_per_subject[i]

        coeffs = A[start_idx:end_idx]
        
        for cat_idx, category in enumerate(category_names):
            # Images with this category
            with_category = label_matrix[start_idx:end_idx, cat_idx] == 1
            
            if np.sum(with_category) > 0 and np.sum(~with_category) > 0:
                for comp_idx in range(n_components):
                    # Calculate mean for images with and without this category
                    mean_with = np.mean(coeffs[with_category, comp_idx])
                    mean_without = np.mean(coeffs[~with_category, comp_idx])
                    
                    # Store the mean for images with this category
                    category_subj_means[n_categories * i + cat_idx, comp_idx] = mean_with
                
                # Calculate the effect size (difference from images without the category)
                # category_subj_effect[cat_idx, i] = mean_with - mean_without
                
                # # Perform t-test
                # t_stat, p_val = stats.ttest_ind(
                #     coeffs[with_category].mean(),
                #     coeffs[~with_category].mean(),
                #     equal_var=False  # Welch's t-test
                # )
                # p_values[cat_idx, i] = p_val

        start_idx = end_idx

    # Create DataFrames for results
    cate_subj_index = ['subj' + str(i+1) + '_' + category for i in range(n_subjects) for category in category_names]
    means_df = pd.DataFrame(
        category_subj_means,
        index=cate_subj_index,
        columns=[f'IC {i+1}' for i in range(n_components)]
    )
    # effect_df = pd.DataFrame(
    #     category_subj_effect,
    #     index=category_names,
    #     columns=[f'Subject {i+1}' for i in range(n_subjects)]
    # )

    # plot heatmap of means_df
    plt.figure(figsize=(10, 18))
    sns.heatmap(means_df, cmap="coolwarm",
                yticklabels=cate_subj_index,
                xticklabels=['IC' + str(i+1) for i in range(n_components)],)
    plt.title("Average Mixing Coefficients per Subject")
    plt.xlabel("Subjects")
    plt.ylabel("Categories")
    plt.tight_layout()
    plt.show()
    # plt.savefig(os.path.join(output_dir, 'avg_mixing_heatmap.png'), dpi=300)
    plt.close()

    # plot heatmap of effect_df
    # plt.figure(figsize=(10, 8))
    # sns.heatmap(effect_df, cmap="RdBu_r", center=0,
    #             xticklabels=[f"Subject {i+1}" for i in range(n_subjects)],
    #             yticklabels=category_names)
    # plt.title("Effect of Category on Mixing Coefficients")
    # plt.xlabel("Subjects")
    # plt.ylabel("Categories")
    # plt.tight_layout()
    # plt.show()
    # # plt.savefig(os.path.join(output_dir, 'category_effects_heatmap.png'), dpi=300)
    # plt.close()


def analyze_2_category(mixing_file, n_components, target_category=('food', 'sports')):
    output_dir = mixing_file.split('/')[-1].split('_loadings.npy')[0]
    output_dir = f'Viz/ica/category/{output_dir}/single_category_subj/'
    os.makedirs(output_dir, exist_ok=True)

    n_subjects = 8
    n_images_per_subject = [1000, 1000, 930, 907, 1000, 930, 1000, 907]
    category_names = ['accessory', 'animal', 'appliance', 'electronic', 'food', 'furniture', 'indoor', 'kitchen', 'outdoor', 'person', 'sports', 'vehicle']
    n_categories = len(category_names)

    label_matrix = np.load('data/NSD/coco_labels_val.npy')
    A = np.load(mixing_file)

    mixing_1 = np.zeros((n_subjects, n_components))
    mixing_2 = np.zeros((n_subjects, n_components))

    with_category_1 = label_matrix[:, category_names.index(target_category[0])] == 1
    with_category_2 = label_matrix[:, category_names.index(target_category[1])] == 1

    start_idx = 0
    for i in range(n_subjects):
        end_idx = start_idx + n_images_per_subject[i]

        coeffs = A[start_idx:end_idx]
        
        mixing_1[i] = np.mean(coeffs[with_category_1[start_idx:end_idx], :], axis=0)
        mixing_2[i] = np.mean(coeffs[with_category_2[start_idx:end_idx], :], axis=0)

        start_idx = end_idx

    diff = np.abs(mixing_1.mean(0) - mixing_2.mean(0))

    # find most different components
    diff_idx = np.argsort(diff)[-5:]  # Get indices of top 3 most different components
    mixing_1_top = mixing_1[np.arange(n_subjects)[:, None], diff_idx]
    mixing_2_top = mixing_2[np.arange(n_subjects)[:, None], diff_idx]

    mixing_top = np.concatenate((mixing_1_top, mixing_2_top), axis=1)
    plt.figure(figsize=(10, 8))
    sns.heatmap(mixing_top, cmap="coolwarm",
                xticklabels=[f"IC {i+1}" for i in diff_idx] + [f"IC {i+1}" for i in diff_idx],
                yticklabels=[f"Subject {i+1}" for i in range(n_subjects)],)
    plt.title(f"Top different components for {target_category[0]} and {target_category[1]}")
    plt.show()


def analyze_img_category_combinations(mixing_file, n_components, min_samples=20, max_combos=10, figsize=(15, 10)):
    output_dir = mixing_file.split('/')[-1].split('_loadings.npy')[0]
    output_dir = f'Viz/ica/category/{output_dir}/multi_category/'
    os.makedirs(output_dir, exist_ok=True)

    n_subjects = 8
    n_images_per_subject = [1000, 1000, 930, 907, 1000, 930, 1000, 907]
    category_names = ['accessory', 'animal', 'appliance', 'electronic', 'food', 'furniture', 'indoor', 'kitchen', 'outdoor', 'person', 'sports', 'vehicle']
    n_categories = len(category_names)

    label_matrix = np.load('data/NSD/coco_labels_val.npy')
    A = np.load(mixing_file)

    """Analyze how components relate to specific combinations of categories (original multi-label)."""
    unique_combinations = {}
    for i, row in enumerate(label_matrix):
        # Convert binary vector to tuple for hashing
        combo = tuple(row)
        if combo in unique_combinations:
            unique_combinations[combo].append(i)
        else:
            unique_combinations[combo] = [i]
    
    # Filter combinations with enough samples
    valid_combinations = {combo: indices for combo, indices in unique_combinations.items() 
                         if len(indices) >= min_samples}
    
    # Sort by frequency
    sorted_combinations = sorted(valid_combinations.items(), 
                               key=lambda x: len(x[1]), 
                               reverse=True)[:max_combos]
    
    print(f"Found {len(valid_combinations)} category combinations with at least {min_samples} samples.")
    print(f"Analyzing top {len(sorted_combinations)} combinations...")
    
    # Prepare data for visualization
    combo_data = []
    
    for combo, indices in sorted_combinations:
        # Get category names in this combination
        combo_cats = [category_names[i] for i, present in enumerate(combo) if present]
        combo_name = " + ".join(combo_cats) if combo_cats else "None"
        
        # Calculate mean component values for this combination
        combo_means = np.mean(A[indices], axis=0)
        
        # Compare to overall mean
        overall_mean = np.mean(A, axis=0)
        combo_effect = combo_means - overall_mean
        
        # Perform t-test for each component
        p_values = []
        for comp_idx in range(n_components):
            other_indices = list(set(range(len(A))) - set(indices))
            t_stat, p_val = stats.ttest_ind(
                A[indices, comp_idx],
                A[other_indices, comp_idx],
                equal_var=False
            )
            p_values.append(p_val)
        
        # Store results
        combo_data.append({
            'combination': combo_name,
            'n_samples': len(indices),
            'means': combo_means,
            'effect': combo_effect,
            'p_values': p_values,
            'indices': indices
        })
    
    # Visualize effects on components
    plt.figure(figsize=figsize)
    
    # Create a matrix of effects
    effect_matrix = np.zeros((len(combo_data), n_components))
    p_value_matrix = np.zeros((len(combo_data), n_components))
    combo_names = []
    
    for i, data in enumerate(combo_data):
        effect_matrix[i] = data['effect']
        p_value_matrix[i] = data['p_values']
        combo_names.append(f"{data['combination']} (n={data['n_samples']})")
    
    # Apply Benjamini-Hochberg FDR correction
    p_flat = p_value_matrix.flatten()
    reject, p_corrected, _, _ = multipletests(p_flat, method='fdr_bh')
    p_corrected = p_corrected.reshape(p_value_matrix.shape)
    
    # Create mask for non-significant values
    significance_mask = p_corrected >= 0.05
    
    # Plot heatmap
    sns.heatmap(effect_matrix, cmap="RdBu_r", center=0, 
                mask=significance_mask,
                xticklabels=[f'IC{i+1}' for i in range(n_components)],
                yticklabels=combo_names)
    plt.title('Effect of Category Combinations on Component Values (Only Significant Effects Shown)')
    plt.tight_layout()
    plt.show()
    plt.savefig(os.path.join(output_dir, 'category_combinations_effects_heatmap.png'), dpi=300)
    plt.close()
    
    # For each combination, find most distinctive components
    print("\nMost Distinctive Components for Each Category Combination:")
    print("="*80)
    with open(os.path.join(output_dir, 'category_combinations_effects.txt'), 'w') as f:
        f.write("\nMost Distinctive Components for Each Category Combination:\n")
        f.write("="*80 + "\n")
    
    for data in combo_data:
        # Get components with significant effects
        sig_comps = [i for i, p in enumerate(data['p_values']) if p < 0.05/n_components]  # Bonferroni
        
        if sig_comps:
            # Sort by absolute effect
            sorted_comps = sorted(sig_comps, key=lambda x: abs(data['effect'][x]), reverse=True)
            top_comps = sorted_comps[:3]
            
            comp_effects = [f"IC{i+1}: {data['effect'][i]:.3f}" for i in top_comps]
            print(f"{data['combination']} (n={data['n_samples']}): {', '.join(comp_effects)}")
        else:
            print(f"{data['combination']} (n={data['n_samples']}): No significant components")

        with open(os.path.join(output_dir, 'category_combinations_effects.txt'), 'a') as f:
            if sig_comps:
                f.write(f"{data['combination']} (n={data['n_samples']}): {', '.join(comp_effects)}\n")
            else:
                f.write(f"{data['combination']} (n={data['n_samples']}): No significant components\n")


def viz_image_by_component(mixing_file, n_components, method='tsne', figsize=(15, 10)):
    output_dir = mixing_file.split('/')[-1].split('_loadings.npy')[0]
    output_dir = f'Viz/ica/category/{output_dir}/'
    os.makedirs(output_dir, exist_ok=True)

    n_subjects = 8
    n_images_per_subject = [1000, 1000, 930, 907, 1000, 930, 1000, 907]
    category_names = ['accessory', 'animal', 'appliance', 'electronic', 'food', 'furniture', 'indoor', 'kitchen', 'outdoor', 'person', 'sports', 'vehicle']
    n_categories = len(category_names)

    label_matrix = np.load('data/NSD/coco_labels_val.npy')
    A = np.load(mixing_file)

    # Choose a subset of most common categories for coloring
    category_counts = np.sum(label_matrix, axis=0)
    top_categories = np.argsort(category_counts)[::-1][:5]  # Top 5 most common categories
    
    # Reduce dimensionality to 2D
    if method.lower() == 'tsne':
        reducer = TSNE(n_components=2, random_state=42)
        embedding = reducer.fit_transform(A)
        title = 't-SNE Visualization of Component Space'
    elif method.lower() == 'umap':
        reducer = umap.UMAP(n_components=2, random_state=42)
        embedding = reducer.fit_transform(A)
        title = 'UMAP Visualization of Component Space'
    else:
        reducer = PCA(n_components=2)
        embedding = reducer.fit_transform(A)
        title = 'PCA Visualization of Component Space'
    
    # Create a plot for each top category
    plt.figure(figsize=figsize)
    
    for i, cat_idx in enumerate(top_categories):
        plt.subplot(2, 3, i+1)
        
        category = category_names[cat_idx]
        has_category = label_matrix[:, cat_idx] == 1
        
        # Plot points
        plt.scatter(embedding[~has_category, 0], embedding[~has_category, 1], 
                   color='lightgray', alpha=0.3, label='Without')
        plt.scatter(embedding[has_category, 0], embedding[has_category, 1], 
                   color=f'C{i}', alpha=0.7, label='With')
        
        plt.title(f'Category: {category}')
        plt.legend()
    
    # Plot the final subplot showing combinations of categories
    plt.subplot(2, 3, 6)
    
    # Create a color map for top categories
    top_categories_with_name = [(category_names[i], np.where(label_matrix[:, i] == 1)[0]) for i in top_categories]
    
    # Plot each combination with a different color
    for i, (combo_name, indices) in enumerate(top_categories_with_name):
        plt.scatter(embedding[indices, 0], embedding[indices, 1], 
                   color=f'C{i}', alpha=0.7, label=f'{combo_name} (n={len(indices)})')
    
    plt.title('Common Category Combinations')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.suptitle(title, y=1.02, fontsize=16)
    plt.show()

    plt.savefig(os.path.join(output_dir, f'component_space_{method}.png'), dpi=300)

def viz_fnc(mixing_file):
    output_dir = mixing_file.split('/')[-1].split('_loadings.npy')[0]
    output_dir = f'Viz/ica/fnc/{output_dir}/'
    os.makedirs(output_dir, exist_ok=True)

    n_subjects = 8
    n_images_per_subject = [1000, 1000, 930, 907, 1000, 930, 1000, 907]

    A = np.load(mixing_file)
    n_components = A.shape[1]

    fnc_matrix = np.corrcoef(A, rowvar=False)
    # Set the diagonal to 0 (self-correlation)
    np.fill_diagonal(fnc_matrix, 0)

    # normalize the matrix to be between -1 and 1
    fnc_matrix = (fnc_matrix - np.min(fnc_matrix)) / (np.max(fnc_matrix) - np.min(fnc_matrix)) * 2 - 1

    plt.figure(figsize=(10, 8))
    sns.heatmap(fnc_matrix,
                cmap='coolwarm',    # Color map (blue-white-red is good for correlations)
                center=0,          # Center the color map at 0 correlation
                square=True,       # Make cells square
                linewidths=.5,     # Add lines between cells
                cbar_kws={"shrink": .7, "label": "Pearson Correlation (r)"}, # Color bar settings
                # vmin=-1, vmax=1     # Set limits of color bar to -1 and 1
            )
    plt.title(f'Functional Network Connectivity (FNC) Matrix ({n_components} Components)')
    plt.xlabel('Independent Component Index')
    plt.ylabel('Independent Component Index')
    plt.tight_layout()
    plt.show()
    plt.savefig(os.path.join(output_dir, 'fnc_matrix.png'), dpi=300)
    plt.close()


if __name__ == "__main__":
    n_comp = 32
    # get_ica(n_comp)
    # analyze_ica(f'Viz/ica/ica_output_{n_comp}.nii.gz', atlas='schaefer_2018_1000_17', n_co mponents=n_comp, threshold=0.05)
    # analyze_ica_mixing(f'Viz/ica/ica_output_{n_comp}_loadings.npy', n_components=n_comp)
    # analyze_img_category(f'Viz/ica/ica_output_{n_comp}_loadings.npy', n_components=n_comp)
    # analyze_img_category_combinations(f'Viz/ica/ica_output_{n_comp}_stdDiv_perSubj_loadings.npy', n_components=n_comp)
    # analyze_subject_category(f'Viz/ica/ica_output_{n_comp}_stdDiv_loadings.npy', n_components=n_comp)
    # viz_image_by_component(f'Viz/ica/ica_output_{n_comp}_loadings.npy', n_components=n_comp, method='tsne')
    # viz_fnc(f'Viz/ica/ica_output_{n_comp}_loadings.npy')

    # analyze_2_category(f'Viz/ica/ica_output_{n_comp}_loadings.npy', n_components=n_comp)
    plot_components(f'Viz/ica/ica_output_{n_comp}.nii.gz', component_idx=[21, 26, 14])