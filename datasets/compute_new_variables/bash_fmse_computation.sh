#!/bin/bash
#SBATCH --job-name=compute_fmse
#SBATCH --partition=compute
#SBATCH --account=bb1153
#SBATCH --cpus-per-task=64
#SBATCH --ntasks=1
#SBATCH -N 1
#SBATCH --time=8:00:00
#SBATCH --mem=128G

# Load conda environment
source /home/b/b381993/miniforge3/etc/profile.d/conda.sh
conda activate my_new_env

# Navigate to the directory with your Python script
cd /home/b/b381993/DeepFate/datasets/compute_new_variables

# Execute the Python script
python main_fmse.py
