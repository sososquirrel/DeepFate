#!/bin/bash
#SBATCH --job-name=direct_precomputed
#SBATCH --partition=compute
#SBATCH --account=bb1153
#SBATCH --cpus-per-task=64
#SBATCH --ntasks=1
#SBATCH -N 1
#SBATCH --time=8:00:00
#SBATCH --mem=128G



####Print config in a .txt in the folder
srun --mem=64G --ntasks=1 --exclusive --nodes=1 /sw/spack-levante/jupyterhub/jupyterhub/bin/python /home/b/b381993/DeepFate/interpretable_features/direct_precomputed_features.py