#!/bin/bash
#SBATCH --job-name="train"
#SBATCH --partition=gpu
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=2G
#SBATCH --account=education-eemcs-msc-cs

# Load modules:
module load 2022r2
module load openmpi
module load miniconda3

# Set conda env:
unset CONDA_SHLVL
source "$(conda info --base)/etc/profile.d/conda.sh"

# Activate conda, run job, deactivate conda
conda activate /scratch/yunhanwang/.conda/envs/pytorch2
srun python stale_train.py --anno_file anet_anno_action_20p2.json ; python stale_inference.py ; python eval.py ; mv output/STALE_best_100_split.pth.tar output/STALE_best_20p2.pth.tar ; \
     python stale_train.py --anno_file anet_anno_action_40p2.json ; python stale_inference.py ; python eval.py ; mv output/STALE_best_100_split.pth.tar output/STALE_best_40p2.pth.tar ; \
     python stale_train.py --anno_file anet_anno_action_60p2.json ; python stale_inference.py ; python eval.py ; mv output/STALE_best_100_split.pth.tar output/STALE_best_60p2.pth.tar ; \
     python stale_train.py --anno_file anet_anno_action_80p2.json ; python stale_inference.py ; python eval.py ; mv output/STALE_best_100_split.pth.tar output/STALE_best_80p2.pth.tar ; \
     python stale_train.py ; python stale_inference.py ; python eval.py ; mv output/STALE_best_100_split.pth.tar output/STALE_best_100p2.pth.tar
conda deactivate
