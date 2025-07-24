#! /bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 24
#SBATCH --mem=96g
#SBATCH -p qTRDGPUH
#SBATCH -A trends53c17
#SBATCH --gres=gpu:A100:1
#SBATCH --exclude=arctrdagn035,arctrdagn037,arctrddgxa001
#SBATCH --oversubscribe

export PATH="/sysapps/ubuntu-applications/miniconda/4.12.0/miniconda3/bin:$PATH"
cd ~/playground/BrainGen/MindEyeV2/src

source activate
conda activate playground

# jupyter nbconvert recon_inference.ipynb --to python
# jupyter nbconvert enhanced_recon_inference.ipynb --to python

model_name="final_subj01_pretrained_40sess_24bsl_50_p_1.5"

# python recon_inference.py --model_name=${model_name} --subj=1 --n_blocks=4 --hidden_dim=4096
# python enhanced_recon_inference.py --model_name=${model_name} --subj=1
python recon_inference_text.py