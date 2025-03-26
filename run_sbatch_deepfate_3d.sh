#!/bin/bash
#SBATCH --job-name=generate_and_process_dataset
#SBATCH --partition=compute
#SBATCH --account=bb1153
#SBATCH --cpus-per-task=64
#SBATCH --ntasks=2
#SBATCH -N 2
#SBATCH --time=8:00:00
#SBATCH --mem=128G


# Set your case name and store path
CASENAME="VERSION_MARCH_NEW_INTERPRETABLE"
STOREPATH='/work/bb1153/b381993/data/' 


#'/work/bb1153/b381993/data/TEST100'
# Construct the path for the new folder
folder_path="$STOREPATH$CASENAME"

# Create the folder
mkdir -p "$folder_path"

# Check if the folder was created
if [ -d "$folder_path" ]; then
    echo "Folder '$CASENAME' created successfully at: $folder_path"
else
    echo "Failed to create folder '$CASENAME' at: $folder_path"
fi


####Print config in a .txt in the folder
srun --mem=64G --ntasks=1 --exclusive --nodes=1 /sw/spack-levante/jupyterhub/jupyterhub/bin/python print_config.py --pathfolder "$folder_path"


#!/bin/bash

# Run the Python script and capture the output
NUMBER_TOTAL_MCS=$(/sw/spack-levante/jupyterhub/jupyterhub/bin/python datasets/script_valid_mcs.py --print) #when using be careful of mode_3d

echo $NUMBER_TOTAL_MCS

# Check if the captured value is an integer
if [[ $NUMBER_TOTAL_MCS =~ ^[0-9]+$ ]]; then
    echo "NUMBER_TOTAL_MCS: $NUMBER_TOTAL_MCS"
else
    echo "Error: NUMBER_TOTAL_MCS is not an integer."
    exit 1
fi

# Calculate half the value (integer division)
half_number_mcs=$((NUMBER_TOTAL_MCS / 2))

echo "half_number_mcs: $half_number_mcs"

# Run the first Python script for .h5 generation and capture the output name for information
#GENERATED_H5_PATH=$(srun /sw/spack-levante/jupyterhub/jupyterhub/bin/python datasets/main_generate_dataset.py --start_index 0 --stop_index 50 --pathfolder "$folder_path" | grep "Generated dataset path" | awk '{print $NF}') & 
srun --mem=64G --ntasks=1 --exclusive --nodes=1 --cpus-per-task=64 /sw/spack-levante/jupyterhub/jupyterhub/bin/python datasets/main_generate_dataset_with_3d.py --start_index 0 --stop_index $half_number_mcs --pathfolder "$folder_path" &
srun --mem=64G --ntasks=1 --exclusive --nodes=1 --cpus-per-task=64 /sw/spack-levante/jupyterhub/jupyterhub/bin/python datasets/main_generate_dataset_with_3d.py --start_index $half_number_mcs --stop_index $NUMBER_TOTAL_MCS --pathfolder "$folder_path"&


wait

echo "************ SUCCESS GENERATION H5 ************"

echo "Generated h5 path: $GENERATED_H5_PATH"

srun --mem=64G --ntasks=1 --exclusive --nodes=1 /sw/spack-levante/jupyterhub/jupyterhub/bin/python datasets/main_fusion.py --pathfolder "$folder_path"

echo "************ SUCCESS FUSION H5 ************"

# Run the second Python script for features computation 
GENERATED_CSV_PATH=$(srun /sw/spack-levante/jupyterhub/jupyterhub/bin/python interpretable_features/precomputed_features.py --pathfolder "$folder_path" | grep "Generated csv path" | awk '{print $NF}')

echo "************ SUCCESS GENERATION CSV ************"


# Split train and test 
srun --mem=64G --ntasks=1 --exclusive --nodes=1 /sw/spack-levante/jupyterhub/jupyterhub/bin/python model/0_split_dataset.py --pathfolder "$folder_path"

echo "************ SUCCESS SPLIT TRAIN TEST CSV ************"

# Train models on the data
mkdir -p "$folder_path/saved_model"
srun --mem=64G --ntasks=1 --exclusive --nodes=1 /sw/spack-levante/jupyterhub/jupyterhub/bin/python models/pipeline_lasso.py --train_data "$folder_path/train_dataset.csv" --test_data "$folder_path/test_dataset.csv" --save_model "$folder_path/saved_model" 

echo "************ SUCCESS TRAIN ************"


mkdir -p "$folder_path/saved_metrics"
srun --mem=64G --ntasks=1 --exclusive --nodes=1 /sw/spack-levante/jupyterhub/jupyterhub/bin/python models/pipeline_lasso.py --train_data "$folder_path/train_dataset.csv" --test_data "$folder_path/test_dataset.csv" --model_save_path "$folder_path/saved_model"  --metrics_save_path "$folder_path/saved_metrics"

echo "************ SUCCESS METRICS ************"
