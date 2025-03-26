import h5py
import numpy as np
import os
import argparse 
import sys
from tqdm import tqdm

# Add the module directory to the system path to import custom modules
module_dir = '/home/b/b381993'
sys.path.append(module_dir)

# Import required modules from DeepFate
import DeepFate
from DeepFate.config import NB_TIMESTEPS, INPUT_VARIABLES, SPACE_WINDOW
from DeepFate.datasets.script_valid_mcs import NUMBER_TOTAL_VALID_MCS

def combine_h5_datasets(input_folder, output_file):
    """
    Combines multiple HDF5 datasets from the input folder into a single output file.

    This function reads individual HDF5 files containing datasets 'X', 'y', and 'z' 
    and combines them into a larger dataset stored in the provided output file. 

    Args:
        input_folder (str): Path to the folder containing the input HDF5 files.
        output_file (str): Path to the output combined HDF5 file.
    """
    
    # Initialize lists to hold combined data
    combined_X = []
    combined_y = []
    combined_z = []

    # Get a list of all HDF5 files in the input folder
    hdf5_files = [f for f in os.listdir(input_folder) if f.endswith('.h5')]
    print('HDF5 files:', hdf5_files)
    
    # Define the full shapes for the datasets
    X_full_shape = (NUMBER_TOTAL_VALID_MCS, NB_TIMESTEPS, len(INPUT_VARIABLES), SPACE_WINDOW['lat_delta_pixels'], SPACE_WINDOW['lon_delta_pixels'])
    y_full_shape = (NUMBER_TOTAL_VALID_MCS, 3) 
    z_full_shape = (NUMBER_TOTAL_VALID_MCS, 8)
    
    # Counter to track position in the combined datasets
    single_count = 0
    
    # Open the output HDF5 file for writing
    with h5py.File(output_file, 'w') as hf:
        # Create datasets in the output file
        X_dataset = hf.create_dataset('X', X_full_shape, chunks=((1,) + tuple(X_full_shape[1:])))
        y_dataset = hf.create_dataset('y', y_full_shape)
        z_dataset = hf.create_dataset('z', z_full_shape)

        # Iterate through each HDF5 file in the input folder
        for file_name in hdf5_files:
            file_path = os.path.join(input_folder, file_name)
            print('Processing file:', file_path)
            
            # Open each HDF5 file for reading
            with h5py.File(file_path, 'r') as f:
                len_file = f['X'].shape[0]
                print('Number of samples in file:', len_file)

                # Process data in batches
                for batch in tqdm(np.array_split(np.arange(len_file), len_file // 128)):
                    X = f['X'][batch]
                    y = f['y'][batch]
                    z = f['z'][batch]
                    len_batch = X.shape[0]
                    
                    # Print batch size and update progress
                    print(f'Batch size: {len(batch)}')
                    print(f'Updating indices: {single_count} to {single_count + len_batch}')
                    
                    # Write the batch data to the output file
                    X_dataset[single_count:single_count + len_batch] = X
                    y_dataset[single_count:single_count + len_batch] = y
                    z_dataset[single_count:single_count + len_batch] = z

                    # Update the counter for the next batch
                    single_count += len_batch

if __name__ == '__main__':
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Script to combine .h5 datasets into one.')
    parser.add_argument('--pathfolder', help='Path to the folder containing the input datasets.', type=str, required=True)
    args = parser.parse_args()

    # Get input and output folder paths
    pathfolder = args.pathfolder
    input_folder = pathfolder
    output_file = os.path.join(pathfolder, 'DEEPFATE_DATASET.h5')

    # Call the function to combine the datasets
    combine_h5_datasets(input_folder, output_file)
