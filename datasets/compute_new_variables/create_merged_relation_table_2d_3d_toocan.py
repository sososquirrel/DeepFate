import pandas as pd
import os
import dask
import xarray as xr
from tqdm import tqdm

import glob
import re
import os
from tqdm import tqdm
import pandas as pd

def get_stamp(file, mode='2d'):
    if mode=='2d':
        result = file.replace('.PW', '_PW')
        stamp='_'.join(result.split('_')[:-1])
    if mode=='3d':
        stamp='_'.join(file.split('_')[:-1])
    return stamp

def get_dict_time_path(pathdir='/fastdata/ka1081/DYAMOND/data/summer_data/SAM-4km/OUT_2D/*PW*', mode='2d'):
    
    all_files = glob.glob(pathdir)
    time_list=[]
    file_stamp_list=[]
    for file in tqdm(all_files):
        #get time
        ds = xr.open_dataset(file)
        day_of_the_year = float(ds.time.values[0])

        #get stamp
        basename = os.path.basename(file)
        file_stamp = get_stamp(basename, mode=mode)

        #add to list
        time_list.append(day_of_the_year)
        file_stamp_list.append(file_stamp)
    
    dict_path_time={'path_dyamond':file_stamp_list, 'time':time_list}

    return pd.DataFrame(dict_path_time)
        


if __name__ == '__main__':

    dict_2d = get_dict_time_path()
    dict_3d = get_dict_time_path(pathdir='/fastdata/ka1081/DYAMOND/data/summer_data/SAM-4km/OUT_3D/*TABS*', mode='3d')
    
    #merge
    merged_df = pd.merge(dict_2d, dict_3d, on='time', how='outer')
    merged_df = merged_df.sort_values('time').reset_index(drop=True)
    
    # Fill missing values using forward fill
    merged_df['path_dyamond_x'] = merged_df['path_dyamond_x'].ffill()
    merged_df['path_dyamond_y'] = merged_df['path_dyamond_y'].ffill()
    
    #rename
    merged_df.rename(columns={'path_dyamond_x': 'path_dyamond_2d', 'path_dyamond_y': 'path_dyamond_3d'}, inplace=True)

    #save
    merged_df.to_csv('/work/bb1153/b381993/data3/data/MERGED_RELATION_3D_2D_ALL_FILES.csv')