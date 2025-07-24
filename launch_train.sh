#!/bin/bash
export PATH="/sysapps/ubuntu-applications/miniconda/4.12.0/miniconda3/bin:$PATH"
cd ~/playground/BrainGen/public_repo

source activate 
conda activate playground
# export CUDA_VISIBLE_DEVICES=0,1,2,3
export NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)  # Get number of available GPUs
export BATCH_SIZE=24  # 21 for multisubject / 24 for singlesubject (orig. paper used 42 for multisubject / 24 for singlesubject)
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

USE_MULTI_SUBJECT=false

IS_IMAGE=true
USE_PRIOR=true
BLURRY_RECON=false
SUBJ=1
HIDDEN_DIM=4096
VIT="BigG"
TRAIN_POOL=false
NUM_SESSIONS=40

if [ "$USE_MULTI_SUBJECT" = true ]; then
    MODEL_NAME="multi"
    MULTI_SUBJECT_FLAG="--multi_subject"
else
    MODEL_NAME="single"
    MULTI_SUBJECT_FLAG="--no-multi_subject"
fi
if [ "$IS_IMAGE" = true ]; then
    IMAGE_NAME="img"
    IMAGE_FLAG="--is_image"
    SCIPR_NAME="Train.py"
else    
    IMAGE_NAME="cap"
    IMAGE_FLAG="--no-is_image"
    SCIPR_NAME="Train_text.py"
fi
if [ "$USE_PRIOR" = true ]; then
    PRIOR_NAME="prior"
    PRIOR_FLAG="--use_prior"
else
    PRIOR_NAME="nopri"
    PRIOR_FLAG="--no-use_prior"
fi
if [ "$BLURRY_RECON" = true ]; then
    BLURRY_NAME="blurry"
    BLURRY_FLAG="--blurry_recon"
else
    BLURRY_NAME="noblur"
    BLURRY_FLAG="--no-blurry_recon"
fi
if [ "$TRAIN_POOL" = true ]; then
    POOL_NAME="-pool"
    TRAIN_POOL_FLAG="--train_pool"
else
    POOL_NAME=""
    TRAIN_POOL_FLAG=""
fi

model_name="${MODEL_NAME}_subj${SUBJ}_${NUM_SESSIONS}sess_ViT-${VIT}-img-${PRIOR_NAME}-${BLURRY_NAME}-${HIDDEN_DIM}${POOL_NAME}-BrainMoEMulti2-2L3F2"
echo model_name=${model_name}

accelerate launch --num_processes=$(($NUM_GPUS * $COUNT_NODE)) --num_machines=$COUNT_NODE --main_process_ip=$MASTER_ADDR --main_process_port=$MASTER_PORT --mixed_precision=fp16 ${SCIPR_NAME} \
 $MULTI_SUBJECT_FLAG \
 $IMAGE_FLAG \
 $PRIOR_FLAG \
 $BLURRY_FLAG \
 --model_name=${model_name} \
 --subj=${SUBJ} \
 --hidden_dim=${HIDDEN_DIM} \
 --batch_size=${BATCH_SIZE} \
 --max_lr=3e-4 \
 --mixup_pct=0.33 \
 --num_epochs=50  \
 --blur_scale=1 \
 --no-use_image_aug \
 --n_blocks=4 \
 --ckpt_interval=999 \
 ${TRAIN_POOL_FLAG} \
 --temp_coeff=5 \
 --cos_scale=.5 \
 --prior_clip_scale=1 \
 --num_sessions=$NUM_SESSIONS \
 --no-wandb_log \
 --no-ckpt_saving \
#  --ckpt_path=checkpoints/single_subj1_40sess_ViT-BigG-img-prior-noblur-4096-BrainMoEMulti2-2L3F2/best_mse.pth \
#  --train_router_only \




