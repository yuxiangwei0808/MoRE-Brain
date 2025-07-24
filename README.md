# BrainGen: Brain-to-Image Generation with Mixture-of-Experts Encoders

This repository contains the implementation for brain-to-image reconstruction using fMRI data with advanced Mixture-of-Experts (MoE) neural architectures. The system converts brain activity patterns into visual images through a multi-stage pipeline involving neural encoding, diffusion model fine-tuning, and image generation.

## 🚀 Overview

BrainGen implements a state-of-the-art approach for reconstructing visual images from fMRI brain signals using:

- **Mixture-of-Experts (MoE) Brain Encoders**: Advanced neural architectures that learn specialized representations of different brain regions
- **Diffusion Model Fine-tuning**: Custom adaptation of Stable Diffusion XL for brain-guided image generation
- **Multi-modal Training**: Support for both image and text conditioning from brain signals
- **Advanced Routing**: Dynamic expert selection and time-dependent routing for optimal reconstruction

We follow the same data preprocessing settings as MindEyeV2 (`https://github.com/MedARC-AI/MindEyeV2`).

## 📋 Workflow

The complete pipeline consists of four main stages:

### 1. **Brain Encoder Training** 
Train neural encoders to map fMRI voxel data to CLIP embedding space:

```bash
# For image-conditioned training
bash launch_train.sh <num_sessions>

# For text-conditioned training  
bash launch_train_text.sh <num_sessions>
```

**Key Files:**
- `Train.py`: Main training script for image-conditioned brain encoders
- `Train_text.py`: Training script for text-conditioned brain encoders
- `launch_train.sh` / `launch_train_text.sh`: Launch scripts with configuration

### 2. **Diffusion Model Fine-tuning**
Fine-tune Stable Diffusion XL with LoRA adapters using brain-derived embeddings:

```bash
bash launch_tune_diffusion.sh
```

**Key Files:**
- `Finetune_diffusion.py`: Fine-tuning script for Stable Diffusion XL
- `launch_tune_diffusion.sh`: Launch script for diffusion fine-tuning

### 3. **Image Reconstruction**
Generate reconstructed images from brain signals using the trained models:

```bash
# Automatically called in launch_tune_diffusion.sh
python Recon_inference.py --model_name_image <image_model> --model_name_text <text_model> [options]
```

**Key Files:**
- `Recon_inference.py`: Main reconstruction inference script
- `pipeline_stable_diffusion_xl.py`: Custom SDXL pipeline with brain routing

### 4. **Image Refinement** (Optional)
Apply post-processing refinement to improve reconstruction quality, as in MindEyeV2:

```bash
python Recon_inference_refiner.py [options]
```

**Key Files:**
- `Recon_inference_refiner.py`: Optional refinement post-processing

## 🏗️ Architecture Components

### Brain Encoders
- **`brain_encoder.py`**: Base brain encoder with ridge regression and transformer backbone
- **`brain_moe_encoder.py`**: Mixture-of-Experts variants including:
  - `BrainMoE`: Standard MoE with expert routing
  - `BrainMoEMulti`: Multi-layer MoE with hierarchical experts
  - `BrainMoEHier`: Hierarchical MoE with meta-experts

### CLIP Encoders
- **`clip_encoders.py`**: 
  - `CLIPImageEncoder`: Image encoding with ViT-L/ViT-BigG support
  - `CLIPTextEncoderDual`: Dual text encoder (ViT-L + ViT-BigG)

### Routing Systems
- **`routers.py`**: Expert routing mechanisms including:
  - `ExpertRouter`: Basic expert selection with load balancing
  - Advanced time-dependent and context-aware routing

### Base Components
- **`base_encoders.py`**: Foundational neural network components
- **`load_data.py`**: Data loading utilities for NSD dataset
- **`final_evaluations.py`**: Comprehensive evaluation metrics

## 🛠️ Model Configurations

### Training Parameters
Key hyperparameters can be configured via command line arguments:

- `--model_name`: Model identifier for checkpointing
- `--subj`: Subject ID (1-8) for single-subject training
- `--multi_subject`: Enable multi-subject training
- `--num_sessions`: Number of training sessions to include
- `--hidden_dim`: Hidden dimension size (default: 4096)
- `--n_blocks`: Number of transformer blocks (default: 4)
- `--use_prior`: Enable diffusion prior training
- `--batch_size`: Training batch size

### Model Variants
The codebase supports multiple encoder architectures:

1. **BrainEncoder**: Standard brain encoder with ridge regression (from MindEyeV2)
2. **BrainMoE**: Single-layer Mixture-of-Experts
3. **BrainMoEMulti**: Multi-layer MoE with configurable expert factors

## 📊 Data Requirements

The system requires the Natural Scenes Dataset (NSD) with the following structure:
```
data/NSD/
├── wds/                    # WebDataset format
│   └── subj0X/
│       ├── train/          # Training sessions
│       └── new_test/       # Test data
├── betas_all_subj0X_fp32_renorm.hdf5  # fMRI voxel data
├── coco_images_224_float16.hdf5       # COCO images
└── coco_captions.json                 # COCO captions
```

## 🚦 Quick Start

1. **Prepare Data**: Ensure NSD data is available in the expected format
2. **Train Brain Encoder**: 
   ```bash
   bash launch_train.sh 40  # Use 40 sessions
   ```
3. **Fine-tune Diffusion Model**:
   ```bash
   bash launch_tune_diffusion.sh
   ```
4. **Reconstruction**: The reconstruction will be automatically performed after fine-tuning

## 📈 Evaluation

The system includes comprehensive evaluation metrics:
- **Perceptual Similarity**: CLIP and LPIPS metrics
- **Pixel-level Accuracy**: MSE, PSNR, SSIM
- **Semantic Consistency**: Feature-space comparisons
- **Human Evaluation**: Subjective quality assessment protocols

Run evaluations using:
```bash
python final_evaluations.py --all_recons_path <path_to_reconstructions> [options]
```

## 🔧 Advanced Configuration

### Distributed Training
The system supports multi-GPU distributed training via Accelerate:
- Automatic GPU detection and distribution
- Mixed precision (FP16) training
- Gradient synchronization across devices

### Expert Routing Options
Configure MoE behavior via:
- `--num_exp_0`: Number of base experts
- `--exp_factor_list`: Expert multiplication factors per layer
- `--capacity_factor_0`: Expert capacity scaling
- `--train_router_only`: Train only routing components (for cross-subject generalization)

### Diffusion Customization
Fine-tune diffusion behavior with:
- `--rank`: LoRA rank for efficient fine-tuning
- `--snr_gamma`: Signal-to-noise ratio gamma
- `--route_image` / `--route_text`: Enable brain-guided routing

## 📝 Citation

If you use this code in your research, please cite:

```bibtex
@article{wei2025more,
  title={MoRE-Brain: Routed Mixture of Experts for Interpretable and Generalizable Cross-Subject fMRI Visual Decoding},
  author={Wei, Yuxiang and Zhang, Yanteng and Xiao, Xi and Wang, Tianyang and Wang, Xiao and Calhoun, Vince D},
  journal={arXiv preprint arXiv:2505.15946},
  year={2025}
}
```

---

**Note**: This implementation builds upon MindEyeV2 and Stable Diffusion XL. Please ensure proper attribution to the original works.
