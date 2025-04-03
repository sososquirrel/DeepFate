import os
from datetime import datetime
import argparse
import pandas as pd
import sys

# Add the necessary directory to the system path to import custom modules
sys.path.append('/home/b/b381993/')

# Import DeepFate-related modules
import DeepFate.datasets.compute_new_variables.config_const
import DeepFate.datasets
import DeepFate.datasets.compute_new_variables
import DeepFate.config

# Import relevant constants from DeepFate config
from DeepFate.config import START_3D_UTC_SEC, END_3D_UTC_SEC

# Import function to generate precomputed datasets
from DeepFate.datasets.generate_precomputed_datasets import precompute_all_mcs

# Set up argument parser for command-line inputs
parser = argparse.ArgumentParser(description='Script to generate .h5 dataset from precomputed MCS data.')
parser.add_argument('--pathfolder', help='Directory path to save the generated dataset.', type=str, required=True)
parser.add_argument('--start_index', help='Start index for batch processing of MCS data.', type=int, required=True)
parser.add_argument('--stop_index', help='Stop index for batch processing of MCS data.', type=int, required=True)
args = parser.parse_args()

if __name__ == '__main__':
    # Retrieve input arguments
    pathfolder = args.pathfolder
    
    # Generate a timestamp to be included in the output filename
    now = datetime.now()
    str_date = now.strftime("%d-%m-%Y-%H%M")
    
    # Create the output filename based on the current date and the specified indices
    NAME_DATASET = f'DEEPFATE_{str_date}_index_from_{args.start_index}_to_{args.stop_index}.h5'

    # Inform the user that the dataset is about to be generated
    print(f'{NAME_DATASET} is about to be generated')

    # Define the full path to the output .h5 file
    path_output_h5 = os.path.join(pathfolder, NAME_DATASET)

    # Load the merged 2D-3D data table from the specified CSV file
    df_merged_2d_3d = pd.read_csv(DeepFate.datasets.compute_new_variables.config_const.MERGED_TABLE)

    # Call the function to precompute MCS data and generate the .h5 dataset
    precompute_all_mcs(
        start_index=args.start_index, 
        stop_index=args.stop_index, 
        path_output_h5_file=path_output_h5,
        df_merged_2d_3d=df_merged_2d_3d,
        mode_3d=True, 
        UTC_3d_start=START_3D_UTC_SEC, 
        UTC_3d_end=END_3D_UTC_SEC
    )

    # Notify the user that the dataset has been successfully generated
    print(f'{NAME_DATASET} has been successfully generated')

    # Print the path to the generated .h5 dataset
    print(f"Generated dataset path: {path_output_h5}")
