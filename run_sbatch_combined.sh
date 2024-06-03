#!/bin/bash
#SBATCH --job-name=combine_data_and_train
#SBATCH --partition=compute
#SBATCH --account=bb1153
#SBATCH --cpus-per-task=64
#SBATCH --ntasks=1
#SBATCH -N 1
#SBATCH --time=8:00:00
#SBATCH --mem=128G


# Set where to find dataset to combine
CASENAME_1="FINAL_VERSION_AUGUST_new"
STOREPATH_1='/work/bb1153/b381993/data/' 

CASENAME_2="FINAL_VERSION_SEPTEMBER_new"
STOREPATH_2='/work/bb1153/b381993/data/' 

# Construct the path for the new folder
folder_path_1="$STOREPATH_1$CASENAME_1"
folder_path_2="$STOREPATH_2$CASENAME_2"

# Set where to store the combined data
CASENAME="FINAL_VERSION_new"
STOREPATH='/work/bb1153/b381993/data/' 

folder_path="$STOREPATH$CASENAME"


# Create the folder
#mkdir -p "$folder_path"

# Check if the folder was created
if [ -d "$folder_path" ]; then
    echo "Folder '$CASENAME' created successfully at: $folder_path"
else
    echo "Failed to create folder '$CASENAME' at: $folder_path"
fi

echo "Printing configuration variables:"
/sw/spack-levante/jupyterhub/jupyterhub/bin/python print_config.py

#cp "$folder_path_1/train_dataset.csv" "$folder_path/train_dataset_1.csv" 
#cp "$folder_path_1/test_dataset.csv" "$folder_path/test_dataset_1.csv"

#cp "$folder_path_2/train_dataset.csv" "$folder_path/train_dataset_2.csv" 
#cp "$folder_path_2/test_dataset.csv" "$folder_path/test_dataset_2.csv" 


srun --mem=64G --ntasks=1 --exclusive --nodes=1 /sw/spack-levante/jupyterhub/jupyterhub/bin/python model/0_bis_combine_data.py --pathfolder "$folder_path"
srun --mem=64G --ntasks=1 --exclusive --nodes=1 /sw/spack-levante/jupyterhub/jupyterhub/bin/python model/1_training_script.py --pathfolder "$folder_path"
srun --mem=64G --ntasks=1 --exclusive --nodes=1 /sw/spack-levante/jupyterhub/jupyterhub/bin/python model/2_evaluate_scripts.py --pathfolder "$folder_path"
srun --mem=64G --ntasks=1 --exclusive --nodes=1 /sw/spack-levante/jupyterhub/jupyterhub/bin/python model/3_prediction_scripts.py --pathfolder "$folder_path"
srun --mem=64G --ntasks=1 --exclusive --nodes=1 /sw/spack-levante/jupyterhub/jupyterhub/bin/python model/4_plot_scripts_1.py --pathfolder "$folder_path"
srun --mem=64G --ntasks=1 --exclusive --nodes=1 /sw/spack-levante/jupyterhub/jupyterhub/bin/python model/5_plot_scripts_2.py --pathfolder "$folder_path"