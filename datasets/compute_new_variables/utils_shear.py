
from utils_path import get_path_dict_from_timesteps
import os
import pandas as pd
import numpy as np
import xarray as xr

from config_const import FOLDER_PATH

#5966.392

def get_shear_index(i, merged_df, root_dict, folder_path, new_shear = 'SHEAR', var='U'):
    dict_all_path = get_path_dict_from_timesteps(i, merged_df, root_dict)
    path_3d_U = os.path.join(root_dict['root_3d_files'], f"{dict_all_path['path_dyamond_3d']}_{var}.nc")
    get_and_save_shear(path_3d_U=path_3d_U, level_1 = 20., level_2= 978.97, level_3=5966.392, folder_path=folder_path, new_shear=new_shear, var=var)



def get_and_save_shear(path_3d_U, level_1 = 20., level_2= 978.97, level_3=2563.87,new_shear='SHEAR',var='U', folder_path = FOLDER_PATH ):

    # Open the dataset
    ds = xr.open_dataset(path_3d_U)

    # Select the variable and the corresponding levels
    level1_data = ds[var].sel(z=level_1)  # Replace 1 with the actual first level
    level2_data = ds[var].sel(z=level_2 )  # Replace 2 with the actual second level
    level3_data = ds[var].sel(z=level_3 )  # Replace 2 with the actual second level

    # Compute the difference
    # xarray handles NaN values automatically, so the result will have NaNs where applicable
    shear =  level2_data-level1_data
    deep_shear = level3_data-level1_data

    # Optionally, you can fill NaNs with zeros (or other values) if needed:
    # difference = difference.fillna(0)

    # Save the result to a new .nc file
    _, filename = os.path.split(path_3d_U)
    shear.to_netcdf(os.path.join(folder_path, filename.replace(f'{var}.nc', f'{new_shear}.nc')))
    deep_shear.to_netcdf(os.path.join(folder_path, filename.replace(f'{var}.nc', f'DEEP{new_shear}.nc')))

    #print(var, new_shear)
    #print(os.path.join(folder_path, filename.replace(f'{var}.nc', f'{new_shear}.nc')))
    #print(os.path.join(folder_path, filename.replace(f'{var}.nc', f'DEEP{new_shear}.nc')))

