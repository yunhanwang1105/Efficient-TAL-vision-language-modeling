#!/bin/bash
#SBATCH --job-name="try"
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=1G
#SBATCH --partition=compute
#SBATCH --account=Education-EEMCS-Courses-CSE3000

# Load modules:
module load 2022r2
module load openmpi
module load miniconda3

# Set conda env:
unset CONDA_SHLVL
source "$(conda info --base)/etc/profile.d/conda.sh"

# Activate conda, run job, deactivate conda
conda activate /scratch/yunhanwang/.conda/envs/pytorch
srun python try.py
conda deactivate
