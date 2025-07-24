import torch
import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms as transforms
from torchvision.transforms import ToPILImage
import os
from skimage import exposure


def adjust_gamma(image, gamma):
    return exposure.adjust_gamma(image, gamma=gamma)

project_name = 'Text+Image-AblateParams'
model_name = 'OutSize_512'

target_idx = [502, 801, 934, 367, 525, 510, 858, 261,  90, 905, 986, 326, 120, 931,
        127,  81, 942,  10,  85, 457, 210,  96, 769, 776, 512, 460, 900, 105,
        749,  19, 217, 552, 647, 853, 596, 535, 910, 636, 864, 400,  47,   2,
        371, 533, 686, 787, 897, 587, 320, 899]
ii = [1, 2, 6, 11, 14, 20, 21, 31, 33, 38]
target_idx = [target_idx[i] for i in ii]

# all_recons_path = os.path.join('evals', '1', project_name, model_name, 'all_recons_val_gamma.pt')
all_recons_path = 'evals/1/Text+Image-Ablate/Exp_level1/all_recons_val.pt'
# all_recons_path = 'evals/mindbridge/mindbridge_subj1/all_recons.pt'
all_recons = torch.load(all_recons_path).squeeze()

test_dataset = np.load(f'data/NSD/processed/test_data_{1}.npz', allow_pickle=True)['arr_0'][()]
all_images = torch.Tensor(test_dataset['image'])

recons, images = all_recons[target_idx], all_images[target_idx]
# recons = recons / 255


def plot_images(images, ncols=10, img_size=224, subplot_size=3):
    """
    Plot a grid of images with improved handling and performance.
    
    Args:
        images: Tensor of images to plot
        ncols: Number of columns in the grid
        img_size: Target size for the images
        subplot_size: Size of each subplot in inches
    """
    n_images = len(images)
    n_rows = n_images // ncols + (n_images % ncols > 0)
    
    # Dynamically adjust figure height based on number of rows
    figsize = (ncols * subplot_size, n_rows * subplot_size)
    fig, axes = plt.subplots(nrows=n_rows, ncols=ncols, figsize=figsize)
    
    # Pre-process images if needed
    resize_transform = transforms.Resize((img_size, img_size))
    
    # Handle case where axes is 1D (only one row)
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(n_images):
        r, c = i // ncols, i % ncols
        
        # Resize only if needed
        img = images[i]
        if img.shape[-1] != img_size:
            img = resize_transform(img)
        img = torch.tensor(adjust_gamma(np.array(img), 2.5))

        # Convert to PIL only once per image
        pil_img = ToPILImage()(img)
        axes[r, c].imshow(pil_img)
        axes[r, c].axis('off')
    
    # Hide unused subplots
    for i in range(n_images, n_rows * ncols):
        r, c = i // ncols, i % ncols
        axes[r, c].axis('off')
        axes[r, c].set_visible(False)
    
    plt.tight_layout()
    plt.show()
    plt.savefig(f'x.png', dpi=300, bbox_inches='tight')
    return fig

        
plot_images(recons, ncols=10)
# plot_images(images, ncols=10)


def plot_saliency_maps(images, saliency_maps, ncols=10, img_size=224, subplot_size=3):
    from pytorch_grad_cam.utils.image import show_cam_on_image
    
    n_images = len(images)
    n_rows = n_images // ncols + (n_images % ncols > 0)
    
    # Dynamically adjust figure height based on number of rows
    figsize = (ncols * subplot_size, n_rows * subplot_size)
    fig, axes = plt.subplots(nrows=n_rows, ncols=ncols, figsize=figsize)
    
    # Pre-process images if needed
    resize_transform = transforms.Resize((img_size, img_size))
    
    # Handle case where axes is 1D (only one row)
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(n_images):
        r, c = i // ncols, i % ncols
        
        # Resize only if needed
        img = images[i]
        if img.shape[-1] != img_size:
            img = resize_transform(img)
        
        img = np.transpose(img.numpy(), (1, 2, 0))
        img = show_cam_on_image(img, saliency_maps[i], use_rgb=True)

        # Convert to PIL only once per image
        # pil_img = ToPILImage()(img)
        axes[r, c].imshow(img)
        axes[r, c].axis('off')
    
    # Hide unused subplots
    for i in range(n_images, n_rows * ncols):
        r, c = i // ncols, i % ncols
        axes[r, c].axis('off')
        axes[r, c].set_visible(False)
    
    plt.tight_layout()
    plt.show()
    plt.savefig(f'figures/saliency_level1_exp3.png', dpi=300, bbox_inches='tight')
    return fig
    
# saliency_maps = np.load('Viz/subj01/layerwise_mask/saliency_maps_level1_exp3.npy')
# plot_saliency_maps(images, saliency_maps, ncols=10)