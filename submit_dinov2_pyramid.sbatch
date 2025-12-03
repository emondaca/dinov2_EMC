#!/bin/bash
#SBATCH --partition=a100_long
#SBATCH --gres=gpu:a100:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=160Gb
#SBATCH --time=20-00:00:00
#SBATCH --job-name=dinov2_pyramid


# Load modules
module load python/3.10.10
module load cuda/12.1

# Activate conda environment
source /gpfs/data/shenlab/aj4718/miniconda3/etc/profile.d/conda.sh
conda activate dinov2

export WANDB_API_KEY=''
export PYTHONPATH=$PYTHONPATH:.

# Print environment info
echo "Python: $(which python)"
echo "Nodes: $SLURM_JOB_NODELIST"
echo "GPUs: $SLURM_JOB_GPUS"
echo "========================================" 

# Run the training script
CONFIG="dinov2/configs/train/vits14_pyramid.yaml"
OUTPUT_DIR="/gpfs/data/shenlab/aj4718/dinov2/logs/vits14_pyramid"

# Create output directory
mkdir -p $OUTPUT_DIR

# Detect number of GPUs
NUM_GPUS=$(nvidia-smi -L | wc -l)
echo "Detected $NUM_GPUS GPUs"

# Using torchrun for distributed training
# It will automatically use the detected number of GPUs via --nproc_per_node
torchrun --nproc_per_node=$NUM_GPUS dinov2/train/train_pyramid.py \
    --config-file $CONFIG \
    --output-dir $OUTPUT_DIR \
    --run_name pyramid_distillation_4gpu
