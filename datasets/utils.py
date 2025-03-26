import tqdm
import numpy as np
import xarray as xr
import pandas as pd
import pandas as pd
import DeepFate
from scipy import signal

from DeepFate.config import PATH_RELATION_TABLE, INPUT_VARIABLES, ROLLING_WINDOW, GRADIENT_THRESHOLD, FRACTION_MAX_END, FRACTION_MAX_START


relation_table = pd.read_csv(PATH_RELATION_TABLE)

def generate_dyamond_file_path_from_utc(utc_time, df_relation_table_UTC):
    df = df_relation_table_UTC[df_relation_table_UTC['UTC_sec']==utc_time]
    assert len(df) == 1
    return df['path_dyamond'].iloc[0]
    
    
def generate_img_seg_file_path_from_utc(utc_time, df_relation_table_UTC):
    df = df_relation_table_UTC[df_relation_table_UTC['UTC_sec']==utc_time]
    assert len(df) == 1
    return df['img_seg_path'].iloc[0]
    

def compute_delta_time(start, end):
    delta_int = (int(end) - int(start)) * 24 #24 hours
    delta_float = (end%1 - start%1)*100*0.5 #
    return np.round(delta_int + delta_float,2)
    
    
def is_mcs_valid(object_mcs,
                  max_area,
                  min_area,
                  duration_max, 
                  duration_min, 
                  lat_max,
                utc_remove=1470873600, #11 august 00:30 UTC time in seconds
                 quality_threshold=11100 #defined by TOOCAN
                ):  
    
    duration_mcs = object_mcs.INT_duration
    
    idx_max_area_mcs = np.argmax(object_mcs.clusters.LC_surfkm2_235K) #before 172Wm2
    max_area_mcs = object_mcs.clusters.LC_surfkm2_235K[idx_max_area_mcs]
    
    t_max_area_mcs = object_mcs.clusters.LC_UTC_time[idx_max_area_mcs]
    t_start_mcs = object_mcs.clusters.LC_UTC_time[0]
    t_end_mcs = object_mcs.clusters.LC_UTC_time[-1]
    
    lat_mcs = object_mcs.INT_latEnd
    
    begin_time_utc = object_mcs.INT_UTC_timeInit
    
    quality = object_mcs.INT_qltyDCS
    
    bool_verdict = (duration_mcs<duration_max) & (duration_mcs>duration_min) & (max_area_mcs<max_area) & (max_area_mcs>min_area) & (lat_mcs<lat_max) &(lat_mcs>-lat_max) & (begin_time_utc>utc_remove) & (quality<=quality_threshold)
    
    
    return bool_verdict

def get_list_valid_mcs(list_object_mcs,
                  max_area,
                  min_area,
                  duration_max, 
                  duration_min, 
                  lat_max):
    
    bool_idx_mcs = [is_mcs_valid(list_object_mcs[i],
                                          max_area,
                                          min_area,
                                          duration_max, 
                                          duration_min, 
                                          lat_max) for i in tqdm.tqdm(range(len(list_object_mcs)))]
    
    output_list = [list_object_mcs[i] for i in range(len(list_object_mcs)) if bool_idx_mcs[i] is True]
    #print('ratio keep datasets', len(output_list) / len(list_object_mcs))
    
    
    return output_list

    
def is_in_lat_lon_window(object_mcs,
                        lat_min,
                        lat_max,
                        lon_min,
                        lon_max,
                        duration_min,
                        area_min=False):
    
    duration_mcs = object_mcs.duration
    
    lat_mcs = object_mcs.INT_latInit
    lon_mcs = object_mcs.INT_lonInit
    
    area_min_mcs = min(object_mcs.clusters.LC_surfkm2_235K)
    
    bool_verdict = (duration_mcs>duration_min) & (lat_mcs<lat_max) & (lat_mcs>lat_min) & (lon_mcs<lon_max) &(lon_mcs>lon_min)
    
    if area_min !=False:
                       bool_verdict = (duration_mcs>duration_min) & (lat_mcs<lat_max) & (lat_mcs>lat_min) & (lon_mcs<lon_max) &(lon_mcs>lon_min) & (area_min_mcs>area_min)
    return bool_verdict



def get_list_in_lat_lon_window_mcs(list_object_mcs,
                        lat_min,
                        lat_max,
                        lon_min,
                        lon_max,
                        duration_min,
                        area_min=False):
    
    bool_idx_mcs = [is_in_lat_lon_window(list_object_mcs[i],
                                        lat_min,
                                        lat_max,
                                        lon_min,
                                        lon_max,
                                        duration_min) for i in tqdm.tqdm(range(len(list_object_mcs)))]
    if area_min!=False:
        bool_idx_mcs = [is_in_lat_lon_window(list_object_mcs[i],
                                        lat_min,
                                        lat_max,
                                        lon_min,
                                        lon_max,
                                        duration_min,
                                        area_min) for i in tqdm.tqdm(range(len(list_object_mcs)))]
    
    output_list = [list_object_mcs[i] for i in range(len(list_object_mcs)) if bool_idx_mcs[i] is True]
    #print('ratio keep datasets', len(output_list) / len(list_object_mcs))
    return output_list
    
        
    
def binary_segmentation_mask_processing(data, label, transparency:bool=False):
    output = data
    #output = np.nan_to_num(data, copy=False, nan=0)
    #label_MCS = output[output.shape[0]//2, output.shape[1]//2]
    if transparency:
            output[output != label] = np.nan
            output[output == label] = 1
    else:
        output[output != label] = 0
        output[output == label] = 1
    return output

def transparency_processing(data):
    output = data
    #output = np.nan_to_num(data, copy=False, nan=0)
    #label_MCS = output[output.shape[0]//2, output.shape[1]//2]
            
    output[output == 0] = np.nan

    return output


def sel_with_keys(path: str, keys: tuple, lat_slice, lon_slice):    
    if keys == ('lat', 'lon'):
        if path.endswith(('BL.nc', 'MID.nc')):
            output = xr.open_dataarray(path).sel(lat=lat_slice, lon=lon_slice).values
        else:
            output = xr.open_dataarray(path).sel(lat=lat_slice, lon=lon_slice).values[0]
    
    elif keys == ('latitude', 'longitude'):
        output = xr.open_dataset(path).cloud_mask.sel(
            latitude=lat_slice, 
            longitude=lon_slice
        ).values[0]
    
    else:
        raise ValueError(f"Unrecognized keys: {keys}")
    
    return output

def open_xarray_rolling_lon(file_path, lat_bary, lon_bary, lat_delta, lon_delta, output_shape:tuple, keys_sel:tuple=('lat', 'lon')):
    
    def crop_center(img,cropx,cropy):
        y,x = img.shape
        startx = x//2-(cropx//2)
        starty = y//2-(cropy//2)    
        return img[starty:starty+cropy,startx:startx+cropx]
    
    min_lat, max_lat = lat_bary-lat_delta, lat_bary+lat_delta
    min_lon, max_lon = lon_bary-lon_delta, lon_bary+lon_delta
    
    lat_slice = slice(lat_bary-lat_delta, lat_bary+lat_delta)
    
    if min_lon<DeepFate.config.MIN_LONGITUDE_DYAMOND:
        
        diff = DeepFate.config.MIN_LONGITUDE_DYAMOND - min_lon
        
        left_slice = slice(DeepFate.config.MIN_LONGITUDE_DYAMOND, max_lon)
        
        left_array  = sel_with_keys(path=file_path, 
                                    keys=keys_sel, lat_slice=lat_slice, 
                                    lon_slice = left_slice)
        
        right_slice = slice(DeepFate.config.MAX_LONGITUDE_DYAMOND - diff, DeepFate.config.MAX_LONGITUDE_DYAMOND)
        
        right_array  = sel_with_keys(path=file_path, 
                            keys=keys_sel, lat_slice=lat_slice, 
                            lon_slice = right_slice)
        
        assert left_array.shape[0] == right_array.shape[0]
        output = np.concatenate([right_array, left_array], axis=1)
    
    elif max_lon>DeepFate.config.MAX_LONGITUDE_DYAMOND:
        
        diff = max_lon - DeepFate.config.MAX_LONGITUDE_DYAMOND
        
        left_slice = slice(DeepFate.config.MIN_LONGITUDE_DYAMOND, DeepFate.config.MIN_LONGITUDE_DYAMOND + diff)
        
        left_array =sel_with_keys(path=file_path, 
                            keys=keys_sel, lat_slice=lat_slice, 
                            lon_slice = left_slice)
        
        
        right_slice = slice(min_lon , DeepFate.config.MAX_LONGITUDE_DYAMOND)
        right_array =sel_with_keys(path=file_path, 
                    keys=keys_sel, lat_slice=lat_slice, 
                    lon_slice = right_slice)
        
        
        assert left_array.shape[0] == right_array.shape[0]
        output = np.concatenate([right_array, left_array], axis=1)
    
    else:
        lon_slice = slice(lon_bary-lon_delta, lon_bary+lon_delta)
        output = sel_with_keys(path=file_path, 
                    keys=keys_sel, lat_slice=lat_slice, 
                    lon_slice = lon_slice)
    
    
    return crop_center(img = output,cropx = output_shape[0],cropy=output_shape[1])


def gaussian_filter_mcs(msc_mask_snapshot, bump):
    
    kernel = bump[:, np.newaxis] * bump[np.newaxis, :]
    output_filtered = signal.fftconvolve(msc_mask_snapshot, kernel[:, :], mode='same')

    
    return output_filtered


def get_single_validity_idx_start_idx_end(mcs_object, rolling_window=ROLLING_WINDOW, gradient_threshold=GRADIENT_THRESHOLD,  fraction_max_end=FRACTION_MAX_END, fraction_max_start=FRACTION_MAX_START, mode_3d=None, UTC_3d_start=None, UTC_3d_end=None):
    
    # Raise ValueError if mode_3d is True but UTC_3d_start or UTC_3d_end is None
    if mode_3d and (UTC_3d_start is None or UTC_3d_end is None):
        raise ValueError("When mode_3d is True, UTC_3d_start and UTC_3d_end cannot be None.")
    
    
    #parameters
    surfaces = mcs_object.clusters.LC_surfkm2_235K
    nt = len(surfaces)-1

    #get the growth rate in time and compute the rolling mean
    dsdt = np.gradient(surfaces)
    rolling_mean_dsdt = np.convolve(dsdt, np.ones(rolling_window), 'same') / rolling_window

    #get valid sequence of strong growth in absolute value
    valid_absolute_growth = np.where(np.abs(rolling_mean_dsdt)>gradient_threshold)[0]

    if len(valid_absolute_growth) in [0,1]:
        return False, None, None

    #get start and end
    idx_start = valid_absolute_growth[0]
    idx_end = valid_absolute_growth[-1]
    
    #check whether start and end area are valid 
    max_surf = np.max(surfaces)
    start_surf, end_surf = surfaces[idx_start], surfaces[idx_end]
    
    #start surface should not be too big
    if ((start_surf>fraction_max_start*max_surf) or (surfaces[-1]>0.5*max_surf)): 
        return False, None, None

    #end surface should be small enough
    while ((end_surf>fraction_max_end*max_surf) and (idx_end not in [nt, None])): 
        idx_end = idx_end + 1
        end_surf = mcs_object.clusters.LC_surfkm2_235K[idx_end]

    #end surface should not be too big           
    if end_surf>fraction_max_end*max_surf: 
        return False, None, None

    if idx_end - idx_start < DeepFate.config.NB_TIMESTEPS - 1 : 
        return False, None, None
    
    if not idx_start<np.argmax(surfaces)<idx_end:
        return False, None, None
    
    if mode_3d:
        cond_start = mcs_object.clusters.LC_UTC_time[idx_start]<UTC_3d_start
        cond_end = mcs_object.clusters.LC_UTC_time[idx_end]>UTC_3d_end

        if cond_start or cond_end:
            return False, None, None

    return True, idx_start, idx_end




def get_validity_lifecycles_start_end(list_valid_mcs, rolling_window=ROLLING_WINDOW, gradient_threshold=GRADIENT_THRESHOLD,  fraction_max_end=FRACTION_MAX_END, fraction_max_start=FRACTION_MAX_START, mode_3d=None, UTC_3d_start=None, UTC_3d_end=None):

    validity_list =[]
    idx_start_list =[]
    idx_end_list =[]
    
    for mcs_object in tqdm.tqdm(list_valid_mcs):
    
        validity, idx_start, idx_end = get_single_validity_idx_start_idx_end(mcs_object, rolling_window=rolling_window, gradient_threshold=gradient_threshold, fraction_max_end=fraction_max_end, fraction_max_start=fraction_max_start, mode_3d=mode_3d, UTC_3d_start=UTC_3d_start, UTC_3d_end=UTC_3d_end)
        
        validity_list.append(validity)
        idx_start_list.append(idx_start)
        idx_end_list.append(idx_end)
        
    return validity_list, idx_start_list, idx_end_list