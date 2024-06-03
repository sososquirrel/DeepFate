import sys
module_dir = '/home/b/b381993'
sys.path.append(module_dir)
import argparse
import h5py
import numpy as np
from tqdm import tqdm
import pandas as pd
import multiprocessing
import os


from DeepFate.datasets.generate_precomputed_datasets import get_specs_mcs, get_z_data
from DeepFate.datasets.utils import get_list_valid_mcs, get_validity_lifecycles_start_end
from DeepFate.datasets.load_toocan_dyamond import load_TOOCAN_DYAMOND
import DeepFate
from DeepFate.config import PATH_TOOCAN_GLOBAL_FILE
from features import *

path = PATH_TOOCAN_GLOBAL_FILE ###path change with new file
list_object_mcs = load_TOOCAN_DYAMOND(path) ##change load_tooca

list_valid_mcs = get_list_valid_mcs(list_object_mcs = list_object_mcs,
                   max_area=DeepFate.config.MCS_SPECS_RANGE['max_area'][1],
                    min_area=DeepFate.config.MCS_SPECS_RANGE['max_area'][0],
                    duration_max = DeepFate.config.MCS_SPECS_RANGE['duration_hour'][1], #now in hours
                    duration_min = DeepFate.config.MCS_SPECS_RANGE['duration_hour'][0],
                    lat_max=DeepFate.config.MAX_LAT_TROPICS)

validitity, start_times, end_times = get_validity_lifecycles_start_end(list_valid_mcs)
list_valid_mcs_2 = [list_valid_mcs[i] for i in range(len(list_valid_mcs)) if validitity[i] is True]
list_start_times = [start_times[i] for i in range(len(list_valid_mcs)) if validitity[i] is True]
list_end_times = [end_times[i] for i in range(len(list_valid_mcs)) if validitity[i] is True]

label_all = np.array([list_valid_mcs_2[i].DCS_number for i in range(len(list_valid_mcs_2))])


def pipeline(index, filename):
    # Reopen the file in the worker process:
    with h5py.File(filename, 'r') as f:
        images_i = f['X'][index]
        specs_i = f['z'][index]

    df = get_all_features_single_mcs(X_images=images_i, 
                                     specs=specs_i, 
                                     list_valid_mcs_2=list_valid_mcs_2, 
                                     label_all=label_all, 
                                     list_start_times=list_start_times)
    return df

def process_data(args):
    i, path_to_h5 = args
    try:
        return pipeline(i, path_to_h5)
    except Exception as e:
        print(f"Error processing data for index {i}: {e}")
        return None

if __name__ == '__main__':

    # Parse arguments from the user
    parser = argparse.ArgumentParser(description='Arguments generating .h5 dataset')
    parser.add_argument('--pathfolder', help='pathfolder', type=str, required=True)
    args = parser.parse_args()

    # Get the folder path from the command line argument
    folder_path = args.pathfolder

    # Search for the .h5 file within the folder
    h5_files = [f for f in os.listdir(folder_path) if f.endswith('.h5')]
    
    if 'DEEPFATE_DATASET.h5' not in h5_files:
        print("No main .h5 files found in the specified folder.")
        sys.exit(1)

    # Assuming there is only one .h5 file in the folder, you can choose the first one
    h5_file = os.path.join(folder_path, 'DEEPFATE_DATASET.h5')
    
    with h5py.File(h5_file, 'r') as f:
        dataset_shape = f['X'].shape

    print("Dataset Shape:", dataset_shape)
    nb_sys = dataset_shape[0]
    start, end = 0, nb_sys
    
    num_processes = multiprocessing.cpu_count()  # Number of available CPU cores
    
    # Create a list of arguments for process_data function, each containing the index and path_to_h5
    args_list = [(i, h5_file) for i in range(start, end)]
    
    with multiprocessing.Pool(processes=num_processes) as pool:
        results = list(tqdm(pool.imap(process_data, args_list), total=end-start))

    # Filter out None results and concatenate the dataframes
    df_all = pd.concat([res for res in results if res is not None], axis=0)
    df_all.to_csv(os.path.join(folder_path,'interpretable_features_data.csv'), index=False)
    
    print('SUCCESS')
    path =os.path.join(folder_path,'interpretable_features_data.csv')
    print(f"Generated csv path: {path}")
