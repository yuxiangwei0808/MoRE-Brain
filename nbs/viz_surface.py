import cortex
from cortex import db
import numpy as np
import nibabel as nib
from nilearn import datasets
from nilearn import surface
from nilearn import plotting
import numpy as np
import os
import gc
np.random.seed(1234)
import matplotlib.pyplot as plt

def normalize_nifti(data, method='minmax', target_range=(0, 1), exclude_zeros=True):
    data_to_normalize = data
    mask = None # Mask indicating which voxels to normalize

    if exclude_zeros:
        mask = data != 0
        data_to_normalize = data[mask]
        # Initialize output array: zeros where mask is False, unchanged elsewhere for now
        normalized_data = np.zeros_like(data, dtype=np.float32)
    else:
        normalized_data = np.zeros_like(data, dtype=np.float32) # Initialize output

    if exclude_zeros and not np.any(mask):
        pass # normalized_data is already all zeros
    else:
        zero_suffix = " (excluding zeros)" if exclude_zeros else ""

        if method == 'minmax':
            min_val = np.min(data_to_normalize)
            max_val = np.max(data_to_normalize)

            if max_val - min_val == 0:
                 raise Exception
            else:
                scaled_data = (data_to_normalize - min_val) / (max_val - min_val)
                # Scale to the target range
                scaled_data = scaled_data * (target_range[1] - target_range[0]) + target_range[0]

                # Place the results back into the correct locations
                if exclude_zeros:
                    normalized_data[mask] = scaled_data.astype(np.float32)
                else:
                    normalized_data = scaled_data.astype(np.float32)

        elif method == 'zscore':
            mean_val = np.mean(data_to_normalize)
            std_val = np.std(data_to_normalize)

            if std_val == 0:
                raise Exception
            else:
                # Apply z-score standardization only to the selected data
                standardized_data = (data_to_normalize - mean_val) / std_val

                # Place the results back into the correct locations
                if exclude_zeros:
                    normalized_data[mask] = standardized_data.astype(np.float32)
                else:
                    normalized_data = standardized_data.astype(np.float32)
        else:
            raise ValueError(f"Invalid normalization method: '{method}'. Choose 'minmax' or 'zscore'.")


    normalized_data = normalized_data.astype(np.float32)
    return normalized_data

def apply_th(attribution_img, threshold):
    tmp_nonzero = attribution_img[attribution_img != 0]
    thresh_value = np.percentile(tmp_nonzero, threshold)
    attribution_img[np.abs(attribution_img) < thresh_value] = 0
    del tmp_nonzero  # Free memory
    return attribution_img

def map2fsaverage(attribution_img, th=0):
    subject_id = 'fsaverage'
    fsaverage_version = 'fsaverage7'
    fsaverage = datasets.fetch_surf_fsaverage(mesh=fsaverage_version)

    # left_surf_mesh = nib.load(fsaverage['pial_left']) # or use 'infl_left', etc.
    # right_surf_mesh = nib.load(fsaverage['pial_right'])

    texture_left = surface.vol_to_surf(
        attribution_img,
        fsaverage.pial_left,
        radius=3.0, # Search radius in mm around each vertex
        interpolation='linear', # How to interpolate volume data ('linear' or 'nearest')
        kind='depth', # Sampling strategy ('ball', 'line', 'depth')
        inner_mesh=fsaverage.white_left, # Use white matter surface for depth sampling
        n_samples=10 # Number of samples along the normal/depth (used if kind='depth' or 'line')
    )

    texture_right = surface.vol_to_surf(
        attribution_img,
        fsaverage.pial_right,
        radius=3.0,
        interpolation='linear',
        kind='depth',
        inner_mesh=fsaverage.white_right,
        n_samples=10
    )

    texture_left = np.nan_to_num(texture_left)  # Replace NaNs with 0
    texture_right = np.nan_to_num(texture_right)
    
    if th:
        texture_left = apply_th(texture_left, 80)
        texture_right = apply_th(texture_right, 80)

    attribution_surface_data = np.concatenate([texture_left, texture_right], dtype=np.float16)
    del texture_left, texture_right, attribution_img
    gc.collect()

    return attribution_surface_data, subject_id

def map2S1(attribution_img):
    subject_id = 'S1'
    s1_pia_left_path = db.get_surf(subject_id, "pia", "left")
    s1_pia_right_path = db.get_surf(subject_id, "pia", "right")
    s1_wm_left_path = db.get_surf(subject_id, "wm", "left")
    s1_wm_right_path = db.get_surf(subject_id, "wm", "right")

    texture_left = surface.vol_to_surf(
        attribution_img,           # Your volume data (Nibabel image object) in S1 space
        s1_pia_left_path,         # Path to S1 left pial surface
        radius=3.0,                       # Search radius in mm
        interpolation='linear',           # Interpolation method
        kind='depth',                     # Sampling strategy ('depth' uses inner/outer surfaces)
        inner_mesh=s1_wm_left_path,       # Path to S1 left white matter surface
        n_samples=10                      # Number of samples along the normal
    )

    texture_right = surface.vol_to_surf(
        attribution_img,
        s1_pia_right_path,        # Path to S1 right pial surface
        radius=3.0,
        interpolation='linear',
        kind='depth',
        inner_mesh=s1_wm_right_path,      # Path to S1 right white matter surface
        n_samples=10
    )
    
    texture_left = np.nan_to_num(texture_left)  # Replace NaNs with 0
    texture_right = np.nan_to_num(texture_right)
    attribution_surface_data = np.concatenate([texture_left, texture_right])

    return attribution_surface_data, subject_id, texture_left, texture_right

def mask_average():
    mask = np.load('Viz/subj01/ridge_grads_mask-layer0.npy')[:1000]
    max_values = np.max(mask, 1)
    max_values[max_values == 0] = 1
    masks = []
    for i in range(mask.shape[1]):
        is_max = (mask[:, i] == max_values)
        masks.append(np.transpose(is_max.astype(np.uint8), (1, 2, 3, 0)))
    del max_values

    for exp_id in range(2):
        name = f'Viz/subj01/layerwise_mask/shap_attrs_level0_bproj_{exp_id}'

        mask = masks[exp_id]

        attribution_img = nib.load(f'{name}.nii.gz')

        affine = attribution_img.affine
        attribution_img = attribution_img.get_fdata()

        assert attribution_img.shape == mask.shape
        attribution_img *= mask
        del mask
        gc.collect()

        nib.save(nib.Nifti1Image(attribution_img.mean(-1), affine=affine), f'Viz/subj01/layerwise_mask/masked-average/shap_attrs_level0_bproj_{exp_id}-mask+mean.nii.gz')

def to_surface():
    root = 'Viz/subj01/layerwise_mask/masked-average-mni'
    out_dir = 'Viz/subj01/layerwise_mask/masked-average-mni-norm-surf'
    os.makedirs(out_dir, exist_ok=True)
    files = os.listdir(root)
    
    for file in files:
        attribution_img = nib.load(os.path.join(root, file))
        affine = attribution_img.affine

        attribution_img = attribution_img.get_fdata()
        attribution_img = normalize_nifti(attribution_img, method='zscore', exclude_zeros=True)
        attribution_img = nib.Nifti1Image(attribution_img, affine=affine)

        attribution_surface_data, subject_id = map2fsaverage(attribution_img)
        # attribution_surface_data, subject_id, texture_left, texture_right = map2S1(attribution_img)

        # vmax = np.percentile(np.abs(attribution_surface_data), 99) # Use 99th percentile of absolute values for symmetric limit
        # vmin = -vmax
        vmax = 1
        vmin = -1
        np.save(os.path.join(out_dir, file.replace('-mask+mean_mni.nii.gz', '.npy')), attribution_surface_data)


    def plot_base():
        attribution_vertex = cortex.Vertex(
            data=attribution_surface_data,
            subject=subject_id,
            cmap='J4R',  # Or 'CyanBlueGrayRedPink', 'J4R', 'Spectral' etc.
            vmin=vmin,
            vmax=vmax,
            # Optional: Add a description
            description="fMRI Beta Attributions"
        )

        # fig = cortex.quickshow(attribution_vertex, with_curvature=True, with_sulci=True, with_labels=True, with_colorbar=True)
        # plt.show()
        cortex.quickflat.make_png(f'shap_attrs_level2_bproj_{exp_id}-mask-ind-th80.png', attribution_vertex, with_curvature=True, with_sulci=True, with_labels=True, with_colorbar=False)


    def plot_rgb():
        num_verts = [len(texture_left), len(texture_right)]

        test1 = attribution_surface_data

        second_verts = [n / 4 for n in num_verts]
        test2 = np.hstack((
            np.abs((texture_left) - second_verts[0]), 
            np.abs((texture_right) - second_verts[1])
        ))

        third_verts = np.random.randint(num_verts[0] + num_verts[1], size=(2,))
        test3 = np.zeros(num_verts[0] + num_verts[1])
        for v in third_verts:
            test3[v-2000: v+2000] = 1

        # Scaling the three datasets to be between 0-255
        test1_scaled = test1 / np.max(test1) * 255
        test2_scaled = test2 / np.max(test2) * 255
        test3_scaled = test3 / np.max(test3) * 255

        # Creating three cortex.Volume objects with the test data as np.uint8
        red = cortex.Vertex(test1_scaled, subject_id)
        green = cortex.Vertex(test2_scaled, subject_id)
        blue = cortex.Vertex(test3_scaled, subject_id)

        # This creates a 2D Vertex object with both of our test datasets for the given subject
        vertex_data = cortex.VertexRGB(red, green, blue, subject_id)
        fig = cortex.quickshow(vertex_data, with_colorbar=False, with_labels=True)
        plt.show()

if __name__ == '__main__':
    to_surface()