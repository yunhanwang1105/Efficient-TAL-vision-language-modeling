#!/bin/bash
#SBATCH --job-name="main"
#SBATCH --partition=gpu
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=2G
#SBATCH --account=Education-EEMCS-Courses-CSE3000

# Load modules:
module load 2022r2
module load openmpi
module load miniconda3

# Set conda env:
unset CONDA_SHLVL
source "$(conda info --base)/etc/profile.d/conda.sh"

# Activate conda, run job, deactivate conda
conda activate /scratch/yunhanwang/.conda/envs/pytorch2
srun python stale_compute_inference.py
conda deactivate
