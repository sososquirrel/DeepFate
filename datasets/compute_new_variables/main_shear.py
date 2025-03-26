import pandas as pd
import os
import dask
from tqdm import tqdm

from utils_shear import get_shear_index
from utils_path import get_corresponding_path_3d, check_existence_in_downloads, longest_consecutive_in_a_row

from config_const import ROOT_DICT_PATHS, FOLDER_PATH, MERGED_TABLE

merged_df = pd.read_csv(MERGED_TABLE)    
corresponding_paths = get_corresponding_path_3d(merged_df)
index_ex = check_existence_in_downloads(corresponding_paths)
longest_interval = longest_consecutive_in_a_row(nums = index_ex)
start_index, end_index = longest_interval[0], longest_interval[-1]


for i in tqdm(range(start_index, end_index, 3)):

    get_shear_index(i=i, 
                    merged_df=merged_df, 
                    root_dict=ROOT_DICT_PATHS, 
                    new_shear = 'SHEAR', 
                    var='U', 
                    folder_path=FOLDER_PATH)
    
    get_shear_index(i=i, 
                    merged_df=merged_df, 
                    root_dict=ROOT_DICT_PATHS, 
                    new_shear = 'SHEARV', 
                    var='V', 
                    folder_path=FOLDER_PATH)
