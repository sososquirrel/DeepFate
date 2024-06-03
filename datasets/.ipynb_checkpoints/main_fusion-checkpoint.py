import h5py
import numpy as np
import os
import argparse 
import sys
# Add module directory to the path
module_dir = '/home/b/b381993'
sys.path.append(module_dir)
from tqdm import tqdm

# Import the required module from DeepFate
import DeepFate
from DeepFate.config import NB_TIMESTEPS, INPUT_VARIABLES, SPACE_WINDOW

from DeepFate.datasets.script_valid_mcs import NUMBER_TOTAL_VALID_MCS


def combine_h5_datasets(input_folder, output_file):
    combined_X = []
    combined_y = []
    combined_z = []

    # Get list of all HDF5 files in the input folder
    hdf5_files = [f for f in os.listdir(input_folder) if f.endswith('.h5')]
    print('hdf5_files', hdf5_files)
    
    X_full_shape = (NUMBER_TOTAL_VALID_MCS, NB_TIMESTEPS, len(INPUT_VARIABLES), SPACE_WINDOW['lat_delta_pixels'], SPACE_WINDOW['lon_delta_pixels'])
    y_full_shape = (NUMBER_TOTAL_VALID_MCS, 3) 
    z_full_shape = (NUMBER_TOTAL_VALID_MCS, 8)
    
    single_count = 0
    
    with h5py.File(output_file, 'w') as hf:
        X_dataset = hf.create_dataset('X', X_full_shape, chunks=((1,) +  tuple(X_full_shape[1:])))
        y_dataset = hf.create_dataset('y', y_full_shape)
        z_dataset = hf.create_dataset('z', z_full_shape)

        for file_name in hdf5_files:
            print('kllkklkl', os.path.join(input_folder, file_name))
            with h5py.File(os.path.join(input_folder, file_name), 'r') as f:
                len_file = f['X'].shape[0]
                print('LEN FILE', len_file)

                for batch in tqdm(np.array_split(np.arange(len_file),len_file//128)):
                    
                    X = f['X'][batch]
                    y = f['y'][batch]
                    z = f['z'][batch]
                    len_batch = X.shape[0]
                    
                    print('BATCH', len(batch))
                    print('single_count, single_count+batch', single_count, single_count+len_batch)
                    
                    X_dataset[single_count:single_count+len_batch] = X
                    y_dataset[single_count:single_count+len_batch] = y
                    z_dataset[single_count:single_count+len_batch] = z

                    single_count += len_batch

if __name__ == '__main__':
    # Parse arguments from the user
    parser = argparse.ArgumentParser(description='Arguments generating .h5 dataset')
    parser.add_argument('--pathfolder', help='pathfolder', type=str, required=True)
    args = parser.parse_args()

    pathfolder = args.pathfolder
    input_folder = pathfolder
    output_file = os.path.join(pathfolder, 'DEEPFATE_DATASET.h5')
    combine_h5_datasets(input_folder, output_file)
