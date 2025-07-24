#!/bin/bash
export PATH="/sysapps/ubuntu-applications/miniconda/4.12.0/miniconda3/bin:$PATH"
cd ~/playground/BrainGen

source activate 
conda activate playground

export NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)  # Get number of available GPUs
export BATCH_SIZE=128  # 21 for multisubject / 24 for singlesubject (orig. paper used 42 for multisubject / 24 for singlesubject)
export GLOBAL_BATCH_SIZE=$((BATCH_SIZE * NUM_GPUS))

# Make sure another job doesnt use same port, here using random number
export MASTER_PORT=$((RANDOM % (19000 - 11000 + 1) + 11000)) 
export HOSTNAMES=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export COUNT_NODE=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | wc -l)
export MASTER_ADDR=localhost
export COUNT_NODE=1
echo MASTER_ADDR=${MASTER_ADDR}
echo MASTER_PORT=${MASTER_PORT}
echo WORLD_SIZE=${NUM_GPUS}

SUBJ=1

accelerate launch --num_processes=$(($NUM_GPUS * $COUNT_NODE)) --num_machines=$COUNT_NODE --main_process_ip=$MASTER_ADDR --main_process_port=$MASTER_PORT --mixed_precision=fp16 Finetune_diffusion.py \
 --model_name_image=single_subj1_40sess_ViT-BigG-img-prior-noblur-4096-BrainMoEMulti2-2L3F2-all+ \
 --model_name_text=single_subj1_40sess_ViT_-cap5-prior-noblur-4096-BigG-BrainMoEMulti2-2L3F2-all+ \
 --no-multi_subject \
 --output_dir=checkpoints/sdxl-finetuned-lora/subj${SUBJ}-MoEMulti2-noMeta-TimeRouterAttn-all+/ \
 --no-wandb_log \
 --rank=16 \
 --snr_gamma=1.5 \
 --route_image \
 --route_text \
 --subj=${SUBJ} \
#  --num_sessions=32
#  --resized_pixel_size=512 \
#  --train_batch_size=16 \
#  --b_size=512

accelerate launch --num_processes=$(($NUM_GPUS * $COUNT_NODE)) --num_machines=$COUNT_NODE --main_process_ip=$MASTER_ADDR --main_process_port=$MASTER_PORT --mixed_precision=fp16 Recon_inference.py \
 --model_name_image=single_subj1_40sess_ViT-BigG-img-prior-noblur-4096-BrainMoEMulti2-2L3F2-all+ \
 --model_name_text=single_subj1_40sess_ViT_-cap5-prior-noblur-4096-BigG-BrainMoEMulti2-2L3F2-all+ \
 --saving_dir=subj${SUBJ}-MoEMulti2-noMeta-TimeRouterAttn-all+ \
 --project_name=Text+Image-mse \
 --lora_ckpt=subj${SUBJ}-MoEMulti2-noMeta-TimeRouterAttn-all+ \
 --subj=${SUBJ} \
#  --random_fmri
#  --b_size=512 \

# accelerate launch --num_processes=$(($NUM_GPUS * $COUNT_NODE)) --num_machines=$COUNT_NODE --main_process_ip=$MASTER_ADDR --main_process_port=$MASTER_PORT --mixed_precision=fp16 Recon_inference_refiner.py \
 --strength=0.2
#  --saving_dir=512-key-Text+Image-trunc \