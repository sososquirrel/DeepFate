import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from tqdm import tqdm
import joblib
from scipy.stats import gaussian_kde
from scipy.stats import percentileofscore
import cartopy.crs as ccrs
import xarray as xr
import sys
module_dir = '/home/b/b381993'
sys.path.append(module_dir)
from DeepFate import config



VARIABLE_PLOT = 'PW'
FULL_NAME_VARIABLE = 'Precipitable Water [mm]'
LATITUDE_SLICE = slice(-40, 40) #degrees
LONGITUDE_SLICE = slice(0, 359)
var = VARIABLE_PLOT
lat_slice = LATITUDE_SLICE
lon_slice=LONGITUDE_SLICE
ROLLING_PIXELS_LON = 4608 #9216/2 corresponds to 180 to be centered to 0°
LON_1, LON_2, LAT_1, LAT_2 = (0.01952, 359.98047, -39.995008, 39.995008)
IMG_EXTENT = ( LON_1-180, LON_2-180, LAT_2, LAT_1)
MODULO_MCS_SEG = 15
COLORBAR_TICKS = np.arange(10,71,1)
CMAP_MCS = plt.cm.get_cmap('gist_ncar', 16)


def get_images_from_mcs_idx(mcs_idx, list_var, with_labels=False):
    mcs, utime_list, lat_list, lon_list, label_mcs = list_valid_mcs_2[mcs_idx], utc_list_all[mcs_idx], lat_list_all[mcs_idx], lon_list_all[mcs_idx], label_all[mcs_idx]
    
    X_all = np.zeros((len(list_var), len(utime_list), 128,128))
    
    for idx_time in range(len(utime_list)):
        path_dyamond_dir_utc_time = generate_dyamond_file_path_from_utc(utime_list[idx_time], df_relation_table_UTC=df_relation_table)
        path_toocan_file_utc_time = generate_img_seg_file_path_from_utc(utime_list[idx_time], df_relation_table_UTC=df_relation_table)
        lat, long =  lat_list[idx_time], lon_list[idx_time] 
        for i_var, var in enumerate(list_var):
            if var =='MCS_segmentation_mask_only' : 
                    file_path = os.path.join(DeepDyamond.config.PATH_SEGMENTED_IMGS, path_toocan_file_utc_time)
                    X = open_xarray_rolling_lon(file_path = file_path, 
                                                lat_bary = lat, 
                                                lon_bary = long, 
                                                lat_delta = lat_delta_degrees, 
                                                lon_delta = lon_delta_degrees, 
                                                output_shape = (lat_delta_pixels, lon_delta_pixels),
                                                keys_sel=('latitude', 'longitude'))
                    
                    X = binary_segmentation_mask_processing(data = X, label = label_mcs, transparency=True)
                    X=np.mod(X,15)
            elif var =='MCS_segmentation' :
                    file_path = os.path.join(DeepDyamond.config.PATH_SEGMENTED_IMGS, path_toocan_file_utc_time)
                    X = open_xarray_rolling_lon(file_path = file_path, 
                                                lat_bary = lat, 
                                                lon_bary = long, 
                                                lat_delta = lat_delta_degrees, 
                                                lon_delta = lon_delta_degrees, 
                                                output_shape = (lat_delta_pixels, lon_delta_pixels),
                                                keys_sel=('latitude', 'longitude'))
                    X[X==0]=np.nan
                    #X[X==1]=label_mcs
                    if not with_labels:
                        X=np.mod(X,15)
                    else:
                        None
                    
            else:
                    file_path = os.path.join(DeepFate.config.PATH_DYAMOND_ROOT_DIR,path_dyamond_dir_utc_time +'.'+var+'.nc')
                    X = open_xarray_rolling_lon(file_path = file_path, 
                            lat_bary = lat, 
                            lon_bary = long, 
                            lat_delta = lat_delta_degrees, 
                            lon_delta = lon_delta_degrees, 
                            output_shape = (lat_delta_pixels, lon_delta_pixels),
                            keys_sel=('lat', 'lon'))
                    
                    
            X_all[i_var, idx_time, :,:]= X
            
    return X_all

def get_data_dyamond_toocan(i:int, path_relation_table = config.PATH_RELATION_TABLE):

    rel_table = pd.read_csv(config.PATH_RELATION_TABLE).sort_values(by='UTC_sec')

    UTC = rel_table.iloc[i]['UTC_sec']
    path_dyamond = rel_table.iloc[i]['path_dyamond']
    img_seg_path = rel_table.iloc[i]['img_seg_path']


    path_file_dyamond_pw = os.path.join(config.PATH_DYAMOND_ROOT_DIR, path_dyamond)+'.'+var+'.2D.nc'
    path_file_toocan = os.path.join(config.PATH_SEGMENTED_IMGS, img_seg_path)
    
    PW_dyamond = xr.open_dataarray(path_file_dyamond_pw).roll(lon=ROLLING_PIXELS_LON).sel(lat=lat_slice, lon=lon_slice)[0]
    PW_np = PW_dyamond.values
    
    img_seg = xr.open_dataset(path_file_toocan).cloud_mask.roll(longitude=4608).sel(latitude = lat_slice, longitude=slice(0,360))[0]
    img_seg_np = img_seg.values
    img_seg_np_mod = np.mod(img_seg_np,MODULO_MCS_SEG)
    
    return PW_np, img_seg_np_mod
