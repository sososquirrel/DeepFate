import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr
import os
import pandas as pd
import DeepFate
import h5py
import tqdm
from multiprocessing import Pool

import warnings
warnings.simplefilter("ignore") 


from DeepFate.datasets.load_toocan_dyamond import load_TOOCAN_DYAMOND
from DeepFate.datasets.utils import get_list_valid_mcs, get_validity_lifecycles_start_end

from DeepFate.datasets.utils import generate_img_seg_file_path_from_utc, generate_dyamond_file_path_from_utc, open_xarray_rolling_lon, binary_segmentation_mask_processing, compute_delta_time


from DeepFate.config import PATH_TOOCAN_GLOBAL_FILE, INPUT_VARIABLES



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


NUMBER_TOTAL_VALID_MCS = len(list_valid_mcs_2)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='print')
    parser.add_argument('--print', action='store_true', help='Print NUMBER_TOTAL_MCS')
    args = parser.parse_args()

    if args.print:
        print(NUMBER_TOTAL_VALID_MCS)