#!/bin/bash

#SBATCH --job-name=nsli
#SBATCH --output=%x.%j.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=1
#SBATCH --account=mx6
#SBATCH --qos=standard
#SBATCH --time=30:00:00

echo "Job started at $(date)"
echo "Hostname: $(hostname)"
echo "GPU Info:"
nvidia-smi

# Enhanced CUDA environment setup (similar to MIT scripts)
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Add CUDA paths like in MIT script
export CUDA_HOME=/mmfs1/home/sp3463/.conda/envs/himenv
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Kill Python processes
kill_python_processes() {
    echo "Killing Python processes on node $(hostname)"
    pkill -9 python
}

# Clean up
kill_python_processes

# Load modules
module purge > /dev/null 2>&1
module load wulver
module load Anaconda3
module load bright
module load gcc/11.2.0

# Activate conda
eval "$(/apps/easybuild/software/Anaconda3/2023.09-0/bin/conda shell.bash hook)"
conda activate himenv

# Install ninja if needed
conda run -n himenv pip install ninja

# Verify GPU and CUDA setup
python -c "
import torch
import os
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'GPU count: {torch.cuda.device_count()}')
print(f'Current device: {torch.cuda.current_device() if torch.cuda.is_available() else \"CPU\"}')
print(f'CUDA_HOME: {os.environ.get(\"CUDA_HOME\", \"Not set\")}')
"

# Configuration
BATCH_SIZE=16
LEARNING_RATE=5e-5
N_EPOCHS=20
LAMBDA_PSL=0.5
SEED=1

# Create checkpoints directory
mkdir -p ./checkpoints


echo "================================================================"
echo "NS-NLI Training"
echo "================================================================"

python train_chunks_fixrev.py


echo "Job completed at $(date)"