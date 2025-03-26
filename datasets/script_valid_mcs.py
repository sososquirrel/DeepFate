import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr
import os
import h5py
import tqdm
import argparse
from multiprocessing import Pool
import warnings
import sys

# Suppress warnings
warnings.simplefilter("ignore")

# Add custom module directory to the system path
sys.path.append('/home/b/b381993/')

# Import functions and configurations from DeepFate
from DeepFate.datasets.load_toocan_dyamond import load_TOOCAN_DYAMOND
from DeepFate.datasets.utils import get_list_valid_mcs, get_validity_lifecycles_start_end
from DeepFate.datasets.utils import generate_img_seg_file_path_from_utc, generate_dyamond_file_path_from_utc, open_xarray_rolling_lon, binary_segmentation_mask_processing, compute_delta_time
from DeepFate.config import PATH_TOOCAN_GLOBAL_FILE, INPUT_VARIABLES, MCS_SPECS_RANGE, MAX_LAT_TROPICS, START_3D_UTC_SEC, END_3D_UTC_SEC

# Path to the TOOCAN global file
path = PATH_TOOCAN_GLOBAL_FILE

# Load the MCS data from the TOOCAN DYAMOND dataset
list_object_mcs = load_TOOCAN_DYAMOND(path)

# Filter the valid MCS based on specified criteria
list_valid_mcs = get_list_valid_mcs(
    list_object_mcs=list_object_mcs,
    max_area=MCS_SPECS_RANGE['max_area'][1],
    min_area=MCS_SPECS_RANGE['max_area'][0],
    duration_max=MCS_SPECS_RANGE['duration_hour'][1],  # duration in hours
    duration_min=MCS_SPECS_RANGE['duration_hour'][0],
    lat_max=MAX_LAT_TROPICS,
)

# Retrieve the validity, start times, and end times of valid MCS
validitity, start_times, end_times = get_validity_lifecycles_start_end(
    list_valid_mcs, 
    mode_3d=True, 
    UTC_3d_start=START_3D_UTC_SEC, 
    UTC_3d_end=END_3D_UTC_SEC
)

# Filter out invalid MCS based on the validity list
list_valid_mcs_2 = [list_valid_mcs[i] for i in range(len(list_valid_mcs)) if validitity[i] is True]
list_start_times = [start_times[i] for i in range(len(list_valid_mcs)) if validitity[i] is True]
list_end_times = [end_times[i] for i in range(len(list_valid_mcs)) if validitity[i] is True]

# Total number of valid MCS
NUMBER_TOTAL_VALID_MCS = len(list_valid_mcs_2)

if __name__ == "__main__":
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(description='Script for handling MCS dataset')
    parser.add_argument('--print', action='store_true', help='Print NUMBER_TOTAL_MCS')
    args = parser.parse_args()

    # If the `--print` argument is provided, print the total number of valid MCS
    if args.print:
        print(NUMBER_TOTAL_VALID_MCS)
