#!/bin/sh
#SBATCH --job-name=create_train_val_test_new_var
#SBATCH --partition=compute
#SBATCH --account=bb1153
#SBATCH --cpus-per-task=32
#SBATCH --ntasks=1
#SBATCH -N 1
#SBATCH --time=8:00:00


### SBATCH --mem=128G
### e.g. request 1 nodes with 1 gpu
### Note: --gres=gpu:x should equal to ntasks-per-node
###SBATCH --nodes=1
###SBATCH --ntasks-per-node=1
###SBATCH --res=
###SBATCH --constraint=p40&gmem24G
###SBATCH --cpus-per-task=256
###SBATCH --mem=10G
###SBATCH --chdir=/scratch/shared/beegfs/your_dir/
###SBATCH --output=/scratch/shared/beegfs/your_dir/%x-%j.out

### change 5-digit MASTER_PORT as you wish, slurm will raise Error if duplicated with others
### change WORLD_SIZE as gpus/node * num_nodes
###export MASTER_PORT=12340
###export WORLD_SIZE=4

### get the first node name as master address - customized for vgg slurm
### e.g. master(gnodee[2-5],gnoded1) == gnodee2
###echo "NODELIST="${SLURM_NODELIST}
###master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
###export MASTER_ADDR=$master_addr
###echo "MASTER_ADDR="$MASTER_ADDR

### init virtual environment if needed
###source ~/anaconda3/etc/profile.d/conda.sh
### the command to run
###srun python main.py

conda env list

srun --mem=128G --ntasks=1 --exclusive --nodes=1 --cpus-per-task=32 /sw/spack-levante/jupyterhub/jupyterhub/bin/python main_generate_dataset.py --casename 'test_filter' --storepath '/work/bb1153/b381993/data/precomputed_datasets_dyamond_new_life_cycles_v3/'


wait
