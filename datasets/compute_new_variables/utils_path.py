import pandas as pd
import os
import dask
import xarray as xr
from tqdm import tqdm
from pathlib import Path
from config_const import ROOT_DICT_PATHS, MERGED_TABLE

def get_corresponding_path_3d(merged_df, 
                              var='U', 
                              path_3d_column_name='path_dyamond_3d'):
    # Filter values in the 'path_dyamond_3d' column and sort by 'UTC'
    corresponding_paths = merged_df.sort_values('time')[path_3d_column_name]
    # Append '_U.nc' to each path
    corresponding_paths = corresponding_paths.apply(lambda x: f"{x}_{var}.nc")
    return corresponding_paths

def check_existence_in_downloads(corresponding_paths, 
                                 root_3d_files = '/fastdata/ka1081/DYAMOND/data/summer_data/SAM-4km/OUT_3D/'):

    index_ex=[]
    for i, value in enumerate(corresponding_paths.values):
        file_path = Path(os.path.join(root_3d_files, value))

        if file_path.exists():
            index_ex.append(i)
        else:
            None
    return index_ex


def longest_consecutive_in_a_row(nums):
    if not nums:
        return []

    # Initialize variables to track the longest sequence
    longest_seq = []
    current_seq = [nums[0]]

    # Iterate over the list and check for consecutive adjacent numbers
    for i in range(1, len(nums)):
        if nums[i] == nums[i - 1] + 1:
            # If current number is consecutive to the previous one, add it to the current sequence
            current_seq.append(nums[i])
        else:
            # If it's not consecutive, compare the current sequence with the longest sequence found so far
            if len(current_seq) > len(longest_seq):
                longest_seq = current_seq
            # Reset the current sequence
            current_seq = [nums[i]]

    # Final check after the loop to update the longest sequence if necessary
    if len(current_seq) > len(longest_seq):
        longest_seq = current_seq

    return longest_seq

def get_path_dict_from_timesteps(i, merged_df, root_dict):
    root_toocan =root_dict['root_toocan']
    root_2d_files = root_dict['root_2d_files']
    root_3d_files = root_dict['root_3d_files']

    merged_df = merged_df.sort_values(by='time')


    UTC = merged_df.iloc[i]['time']
    #str_code = merged_df.iloc[i]['str_code']
    path_dyamond_3d = merged_df.iloc[i]['path_dyamond_3d']
    path_dyamond_2d = merged_df.iloc[i]['path_dyamond_2d']
    #img_seg_path = merged_df.iloc[i]['img_seg_path']

    dict_all_path = {'time':UTC, 
                     #'str_code':str_code, 
                     'path_dyamond_3d':path_dyamond_3d, 
                     'path_dyamond_2d':path_dyamond_2d, 
                     #'img_seg_path':img_seg_path}
    }

    return dict_all_path


if __name__ == "__main__":
    root_dict = ROOT_DICT_PATHS
    merged_df = pd.read_csv(MERGED_TABLE)

    corresponding_paths = get_corresponding_path_3d(merged_df)
    index_ex = check_existence_in_downloads(corresponding_paths)
    longest_interval = longest_consecutive_in_a_row(nums = index_ex)
    start_index, end_index = longest_interval[0], longest_interval[-1]

    print(start_index, end_index)

    dict_all_path = get_path_dict_from_timesteps(i=start_index, merged_df=merged_df, root_dict=root_dict)