#!/bin/bash
#SBATCH --job-name=metrics_models
#SBATCH --partition=compute
#SBATCH --account=bb1153
#SBATCH --cpus-per-task=64
#SBATCH --ntasks=3
#SBATCH -N 1  # Use only 1 node
#SBATCH --time=8:00:00
#SBATCH --mem=128G

# Load any required modules (adjust as needed)
module load python/3.10                     # Replace with the correct Python module if needed
module load h5py                            # If h5py module is available on the cluster

source /home/b/b381993/miniforge3/activate
conda activate my_new_env

# Run each Python file in parallel on separate tasks
srun --ntasks=1 python metrics_models.py &
srun --ntasks=1 python metrics_models_rf.py &
srun --ntasks=1 python metrics_models_mlp.py &

# Wait for all tasks to complete
wait

echo "All models have been trained successfully."
