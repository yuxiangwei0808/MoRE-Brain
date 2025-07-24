from tqdm import tqdm
import nibabel as nib
from scipy.stats import ttest_1samp
import scipy.stats as stats
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from nilearn.image import index_img
from scipy.ndimage import gaussian_filter

def create_prf_matrix(prf_params, visual_field_size=425, vf_extent=10):
    """Create pRF design matrix K from notebook concept"""
    x = np.linspace(-vf_extent, vf_extent, visual_field_size)
    y = np.linspace(-vf_extent, vf_extent, visual_field_size)
    xx, yy = np.meshgrid(x, y)
    
    # Convert polar to Cartesian coordinates for all voxels at once
    angles = np.deg2rad(prf_params['angle'])
    ecc = prf_params['ecc']
    x_centers = ecc * np.cos(angles)
    y_centers = ecc * np.sin(angles)
    
    # Create Gaussian PRFs for all voxels simultaneously
    X = xx[np.newaxis, :, :] - x_centers[:, np.newaxis, np.newaxis]
    Y = yy[np.newaxis, :, :] - y_centers[:, np.newaxis, np.newaxis]
    sigmas = prf_params['sigma'][:, np.newaxis, np.newaxis]
    
    K = np.exp(-(X**2 + Y**2) / (2 * sigmas**2))
    return K.reshape(K.shape[0], -1)  # Flatten spatial dimensions

class PRFProcessor:
    def __init__(self, prf_data_dir='data/NSD/prf/', num_pixel=224, r2_threshold_percentile=75):
        """Initialize PRF processor with pre-calculated matrices"""
        # Load pRF data (only need to do once)
        self.prf_angle = nib.load(f'{prf_data_dir}/prf_angle.nii.gz').get_fdata()
        self.prf_ecc = nib.load(f'{prf_data_dir}/prf_eccentricity.nii.gz').get_fdata()
        self.prf_size = nib.load(f'{prf_data_dir}/prf_size.nii.gz').get_fdata()
        self.prf_r2 = nib.load(f'{prf_data_dir}/prf_R2.nii.gz').get_fdata()
        
        self.num_pixel = num_pixel
        self.brain_shape = self.prf_angle.shape
        
        # Calculate r2 threshold once
        self.r2_threshold = np.percentile(self.prf_r2[self.prf_r2 > 0], r2_threshold_percentile)
        
        # Create base r2 mask (will be combined with beta-specific masks later)
        self.r2_mask = (self.prf_r2 > self.r2_threshold)
        
        # Store K matrix for each voxel that passes the r2 threshold
        # We'll extract the relevant parts based on beta masks later
        self._precalculate_all_valid_voxels()
    
    def _precalculate_all_valid_voxels(self):
        """Precalculate K matrix for all voxels that pass the r2 threshold"""
        valid_voxels = np.argwhere(self.r2_mask)
        
        # Store indices for faster lookup later
        self.valid_indices = valid_voxels
        
        # Extract parameters for valid voxels
        self.prf_params = {
            'angle': self.prf_angle[self.r2_mask],
            'ecc': self.prf_ecc[self.r2_mask],
            'sigma': self.prf_size[self.r2_mask],
            'r2': self.prf_r2[self.r2_mask]
        }
        
        # Precalculate K matrix for all r2-valid voxels
        self.K_all_valid = create_prf_matrix(self.prf_params, self.num_pixel)
        
        # Track which indices in K_all_valid correspond to which voxels in brain space
        self.idx_to_brain_pos = {i: tuple(pos) for i, pos in enumerate(valid_voxels)}
        self.brain_pos_to_idx = {tuple(pos): i for i, pos in enumerate(valid_voxels)}
    
    def process_beta_map(self, beta_map, beta_threshold=2.0, use_ttest=False, visualize=True, saving=False):
        """Process a single beta map to create a saliency map"""
        
        # Create combined mask using r2 and beta threshold
        if use_ttest:
            _, pvals = ttest_1samp(beta_map, 0)
            beta_threshold = stats.t.ppf(0.95, df=beta_map.shape[-1]-1)
        
        beta_mask = (beta_map > beta_threshold)
        combined_mask = self.r2_mask

        assert np.sum(combined_mask) > 0, "No voxels passed the combined mask"
        
        # Get beta values for voxels that pass both thresholds
        beta_values = beta_map[combined_mask]
        r2_values = self.prf_r2[combined_mask]
        
        # Find which indices in our precalculated K matrix we need
        valid_positions = np.argwhere(combined_mask)
        valid_indices = [self.brain_pos_to_idx.get(tuple(pos)) for pos in valid_positions 
                         if tuple(pos) in self.brain_pos_to_idx]
        
        # Extract relevant rows from the K matrix
        K_subset = self.K_all_valid[valid_indices]
        
        # Weight K matrix by activation and model fit
        weights = beta_values * r2_values
        weighted_K = K_subset * weights[:, np.newaxis]
        
        # Sum across voxels to create saliency map
        saliency_map = weighted_K.sum(axis=0).reshape(self.num_pixel, self.num_pixel)
        
        # Normalize
        if np.max(saliency_map) > np.min(saliency_map):  # Avoid division by zero
            saliency_map = (saliency_map - saliency_map.min()) / (saliency_map.max() - saliency_map.min())
        
        saliency_map = gaussian_filter(saliency_map, sigma=1)
        
        if visualize:
            plt.figure(figsize=(8, 8))
            plt.imshow(saliency_map, origin='lower', extent=[-10, 10, -10, 10])
            plt.colorbar(label='Saliency')
            plt.title('pRF-weighted Saliency Map')
            plt.show()
        
        return saliency_map
    
    def process_multiple_beta_maps(self, beta_maps, beta_threshold=2.0, use_ttest=False, visualize=False, saving=False, exp_id=0):
        """Process multiple beta maps and return a list of saliency maps"""
        saliency_maps = []
        mask = np.load('Viz/subj01/ridge_grads_mask-layer1.npy', mmap_mode='r')[:1000]
        
        for i in tqdm(range(beta_maps.shape[-1])):
            print(f"Processing beta map {i+1}/{1000}")
            beta_map = index_img(beta_maps, i).get_fdata()

            m = mask[i, exp_id]
            beta_map *= m

            saliency_map = self.process_beta_map(
                beta_map, 
                beta_threshold=beta_threshold,
                use_ttest=use_ttest,
                visualize=visualize,
                saving=saving
            )
            saliency_maps.append(saliency_map)

            if i % 10 == 1:
                np.save('Viz/subj01/overall/saliency_maps.npy', np.stack(saliency_maps, 0))
            
        return saliency_maps

if __name__ == "__main__":
    # Initialize the processor (loads data and precalculates K matrix)
    processor = PRFProcessor(num_pixel=224)

    level_id = 1
    exp_id = 2
    
    beta_maps = nib.load(f'Viz/subj01/layerwise_mask/shap_attrs_level{level_id}_bproj_{exp_id}.nii.gz')
    saliency_maps = processor.process_multiple_beta_maps(beta_maps)

    np.save(f'Viz/subj01/layerwise_mask/saliency_maps_level{level_id}_exp{exp_id}.npy', np.stack(saliency_maps, 0))