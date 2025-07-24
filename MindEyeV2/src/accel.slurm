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

export NUM_GPUS=1  # Set to equal gres=gpu:#!
export BATCH_SIZE=21 # 21 for multisubject / 24 for singlesubject (orig. paper used 42 for multisubject / 24 for singlesubject)
export GLOBAL_BATCH_SIZE=$((BATCH_SIZE * NUM_GPUS))

# Make sure another job doesnt use same port, here using random number
export MASTER_PORT=$((RANDOM % (19000 - 11000 + 1) + 11000)) 
export HOSTNAMES=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export COUNT_NODE=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | wc -l)
echo MASTER_ADDR=${MASTER_ADDR}
echo MASTER_PORT=${MASTER_PORT}
echo WORLD_SIZE=${COUNT_NODE}

model_name="singlesubject_subj01_40sess_prior"
accelerate launch --num_processes=$(($NUM_GPUS * $COUNT_NODE)) --num_machines=$COUNT_NODE --main_process_ip=$MASTER_ADDR --main_process_port=$MASTER_PORT --mixed_precision=fp16 Train.py --model_name=${model_name} --no-multi_subject --subj=1 --batch_size=${BATCH_SIZE} --max_lr=3e-4 --mixup_pct=.33 --num_epochs=150 --use_prior --prior_scale=30 --clip_scale=1 --no-blurry_recon --blur_scale=.5 --no-use_image_aug --n_blocks=4 --hidden_dim=1024 --num_sessions=40 --ckpt_interval=999 --ckpt_saving --wandb_log

# model_name="multisubject_CSI4_meanTest"
# echo model_name=${model_name}
# accelerate launch --num_processes=$(($NUM_GPUS * $COUNT_NODE)) --num_machines=$COUNT_NODE --main_process_ip=$MASTER_ADDR --main_process_port=$MASTER_PORT --mixed_precision=fp16 Train_bold5000.py \
#     --data_path=/data/users1/ywei/data/NSD \
#     --cache_dir=/data/users1/ywei/data/NSD/.cache \
#     --model_name=${model_name} \
#     --batch_size=${BATCH_SIZE} \
#     --max_lr=3e-4 \
#     --mixup_pct=.33 \
#     --num_epochs=150 \
#     --use_prior \
#     --prior_scale=30 \
#     --clip_scale=1 \
#     --no-blurry_recon \
#     --blur_scale=.5 \
#     --no-use_image_aug \
#     --n_blocks=4 \
#     --hidden_dim=1024 \
#     --ckpt_interval=10 \
#     --ckpt_saving \
#     --wandb_log \
#     --subj_list CSI1 CSI2 CSI3 CSI4 \
#     --target_subj=CSI4 \
    # --no-multi_subject \

# singlesubject finetuning
#model_name="finetuned_subj01_40sess"
#echo model_name=${model_name}
#accelerate launch --num_processes=$(($NUM_GPUS * $COUNT_NODE)) --num_machines=$COUNT_NODE --main_process_ip=$MASTER_ADDR --main_process_port=$MASTER_PORT --mixed_precision=fp16 Train.py --data_path=/weka/proj-fmri/shared/mindeyev2_dataset --cache_dir=/weka/proj-fmri/shared/cache --model_name=${model_name} --no-multi_subject --subj=1 --batch_size=${BATCH_SIZE} --max_lr=3e-4 --mixup_pct=.33 --num_epochs=150 --use_prior --prior_scale=30 --clip_scale=1 --no-blurry_recon --blur_scale=.5 --no-use_image_aug --n_blocks=4 --hidden_dim=1024 --num_sessions=40 --ckpt_interval=999 --ckpt_saving --wandb_log --multisubject_ckpt=../train_logs/multisubject_excludingsubj01_40sess
