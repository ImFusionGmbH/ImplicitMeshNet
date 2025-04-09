#!/bin/zsh
# This script schedules a training
#SBATCH --job-name=implicit_mesh_net_train
#SBATCH --time=3-23:59:59
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=6000Ada
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --output=/N/laves/implicitmeshnet/slurm_logs/slurm_log_%j_stdout.log
#SBATCH --error=/N/laves/implicitmeshnet/slurm_logs/slurm_log_%j_stderr.log

set -e  # Exit on error

source $HOME/.zshrc

cd /home/laves/implicitmeshnet
git reset --hard HEAD
git pull
source env/bin/activate

export PATH=/usr/local/cuda-12.8/bin:/usr/lib/gcc/x86_64-linux-gnu/11:$PATH

srun python deep_representation_pretrain.py \
    --device=cuda \
    --num_stages=4 \
    --num_epochs=1000

srun python implicitmeshnet_train.py \
    --device=cuda \
    --mesh2voxel_config_path=/N/laves/implicitmeshnet/configs_mesh2voxel/2025-03-25_22-41-12.json \
    --data_dir=/N/laves/implicitmeshnet/mmwhs_train \
    --num_epochs=2000 \
    --dataset_repeat=6 \
    --use_labels="2," \
    --target_spacing="1.3,1.3,1.3" \
    --crop_size="128,128,128" \
    --mesh2voxel_volume_size=128 \
    --mesh2voxel_num_stages=4 \
    --surface_loss_weight=0.1 \
    --laplacian_loss_weight_max_epoch=800 \
    --artifacts_dir=/N/laves/implicitmeshnet

srun python implicitmeshnet_train.py \
    --device=cuda \
    --mesh2voxel_config_path=/N/laves/implicitmeshnet/configs_mesh2voxel/2025-03-25_22-41-12.json \
    --data_dir=/N/laves/implicitmeshnet/mmwhs_train \
    --num_epochs=2000 \
    --dataset_repeat=6 \
    --use_labels="3," \
    --target_spacing="1.3,1.3,1.3" \
    --crop_size="128,128,128" \
    --mesh2voxel_volume_size=128 \
    --mesh2voxel_num_stages=4 \
    --surface_loss_weight=0.1 \
    --laplacian_loss_weight_max_epoch=800 \
    --artifacts_dir=/N/laves/implicitmeshnet
