import sys
import os
from datetime import datetime
import argparse

# Add module directory to the path
module_dir = '/home/b/b381993'
sys.path.append(module_dir)

# Import the required module from DeepFate
import DeepFate
from DeepFate.datasets.generate_precomputed_datasets import precompute_all_mcs

# Set up command-line argument parsing
parser = argparse.ArgumentParser(description='Script for generating .h5 dataset from precomputed MCS data.')
parser.add_argument('--pathfolder', help='Directory path to save the generated dataset.', type=str, required=True)
parser.add_argument('--start_index', help='Start index for the batch processing of MCS data.', type=int, required=True)
parser.add_argument('--stop_index', help='End index for the batch processing of MCS data.', type=int, required=True)
args = parser.parse_args()

if __name__ == '__main__':

    # Retrieve command-line arguments
    pathfolder = args.pathfolder
    
    # Get the current timestamp to include in the output filename
    now = datetime.now()
    str_date = now.strftime("%d-%m-%Y-%H%M")
    
    # Create the output dataset file name based on the current timestamp and indices
    NAME_DATASET = f'DEEPFATE_{str_date}_index_from_{args.start_index}_to_{args.stop_index}.h5'

    print(f'{NAME_DATASET} is about to be generated')

    # Define the full path to the output .h5 file
    path_output_h5 = os.path.join(pathfolder, NAME_DATASET)

    # Call the function to precompute the MCS data and generate the .h5 file
    precompute_all_mcs(start_index=args.start_index, stop_index=args.stop_index, path_output_h5_file=path_output_h5)

    # Confirmation message after the dataset is generated
    print(f'{NAME_DATASET} has been successfully generated')

    # Print the path to the generated .h5 file
    print(f"Generated dataset path: {path_output_h5}")
