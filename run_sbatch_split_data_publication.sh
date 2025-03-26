#!/bin/bash
#SBATCH --job-name=split_deepfate_for_archive
#SBATCH --partition=compute
#SBATCH --account=bb1153
#SBATCH --cpus-per-task=64
#SBATCH --ntasks=1
#SBATCH -N 1
#SBATCH --time=8:00:00
#SBATCH --mem=128G

# Load any required modules (adjust as needed)
module load python/3.10                     # Replace with the correct Python module if needed
module load h5py                            # If h5py module is available on the cluster

source /home/b/b381993/miniforge3/activate
conda activate my_new_env

# Run the Python script
srun /sw/spack-levante/jupyterhub/jupyterhub/bin/python /home/b/b381993/DeepFate/split_h5_for_data_publication.py
