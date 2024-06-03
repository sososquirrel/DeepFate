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

########## FUNCTIONS FOR BUILD ITEM


def get_specs_mcs(object_mcs, start_time, end_time):
    
    nb_timesteps = DeepFate.config.NB_TIMESTEPS # 2 but previously 10
    
    if (end_time - start_time)<(nb_timesteps-1):
        print('ERROR no')
    
    if len(object_mcs.clusters.LC_UTC_time) < nb_timesteps:
        raise ValueError(' len(object_mcs.clusters.LC_UTC_time) <= nb_timestep')
    
    if len(object_mcs.clusters.LC_lon) < nb_timesteps:
        raise ValueError(' len(object_mcs.clusters.LC_UTC_time) <= nb_timestep')
    
    duration_mcs = (int(end_time) - int(start_time) + 1 )*0.5 # in hours
    idx_max = np.argmax(object_mcs.clusters.LC_surfkm2_235K)
    if not start_time<idx_max<end_time:
        print('ERROR', start_time, idx_max, end_time)
        raise ValueError('max not in start -> end')
        
    max_extend = np.sqrt(np.max(object_mcs.clusters.LC_surfkm2_235K)) #before 172
    
    idx_max_extension = np.argmax(object_mcs.clusters.LC_surfkm2_235K)
    delta_time_max_extension = (int(idx_max_extension) - int(start_time))/2
    
    dict_specs_mcs = {'label_toocan_mcs':object_mcs.DCS_number,
                        'utc_list_mcs':object_mcs.clusters.LC_UTC_time[start_time:start_time+nb_timesteps],
                        'lat_list_mcs':object_mcs.clusters.LC_lat[start_time:start_time+nb_timesteps],
                        'lon_list_mcs':object_mcs.clusters.LC_lon[start_time:start_time+nb_timesteps],
                        'surf_km_list_mcs':object_mcs.clusters.LC_surfkm2_235K[start_time:start_time+nb_timesteps],
                        'duration_mcs':duration_mcs,
                        'average_velocity':object_mcs.clusters.LC_velocity[start_time:start_time+nb_timesteps],
                        'classif_JIRAK' : object_mcs.INT_classif_JIRAK,
                        'dist' : object_mcs.INT_distance,
                      'max_extend' : max_extend,
                      'delta_time_maximal_extension' : delta_time_max_extension
                     }
    
    
    #/!\ est-ce du debug ?
    #if object_mcs.DCS_number<100:
        #raise ValueError(f'{dict_specs_mcs}')
    assert len(set([len(dict_specs_mcs['utc_list_mcs']),
                   len(dict_specs_mcs['lat_list_mcs']),
                   len(dict_specs_mcs['lon_list_mcs']),
                   len(dict_specs_mcs['surf_km_list_mcs'])])) == 1

    

    return dict_specs_mcs
    

def get_X_data(label_mcs, utc_list, lat_list, long_list, df_relation_table):
    
    nb_timesteps = DeepFate.config.NB_TIMESTEPS
    input_variables = DeepFate.config.INPUT_VARIABLES
    
    if len(utc_list) != nb_timesteps:
        print('ERROR -> len(utc_list)', len(utc_list), 'nb_timesteps', nb_timesteps)
        raise ValueError('len(utc_list) != nb_timesteps')
    if len(lat_list) != nb_timesteps:
        print('ERROR -> len(utc_list)', len(utc_list), 'nb_timesteps', nb_timesteps)
        raise ValueError('len(utc_list) != nb_timesteps')
    
    lat_delta_pixels = DeepFate.config.SPACE_WINDOW['lat_delta_pixels']
    lon_delta_pixels = DeepFate.config.SPACE_WINDOW['lon_delta_pixels']
    lat_delta_degrees = DeepFate.config.SPACE_WINDOW['lat_delta_degrees']
    lon_delta_degrees = DeepFate.config.SPACE_WINDOW['lon_delta_degrees']
    
    X_data = np.zeros((nb_timesteps, len(input_variables),lon_delta_pixels, lat_delta_pixels))
    
    for i, (utc_time, lat, long) in enumerate(zip(utc_list, lat_list, long_list)):
        
        path_dyamond_dir_utc_time = generate_dyamond_file_path_from_utc(utc_time = utc_time,
                                                                        df_relation_table_UTC = df_relation_table)
        

        
        path_toocan_file_utc_time = generate_img_seg_file_path_from_utc(utc_time = utc_time,
                                                                   df_relation_table_UTC = df_relation_table)
        
        for j, var in enumerate(input_variables):
            if var =='MCS_segmentation' : 
                file_path = os.path.join(DeepFate.config.PATH_SEGMENTED_IMGS, path_toocan_file_utc_time) #path segmented images : new file
                
                X = open_xarray_rolling_lon(file_path = file_path, 
                                            lat_bary = lat, 
                                            lon_bary = long, 
                                            lat_delta = lat_delta_degrees, 
                                            lon_delta = lon_delta_degrees, 
                                            output_shape = (lat_delta_pixels, lon_delta_pixels),
                                            keys_sel=('latitude', 'longitude'))
                #X = binary_segmentation_mask_processing(data = X, label = label_mcs)
            
            else:
                file_path = os.path.join(DeepFate.config.PATH_DYAMOND_ROOT_DIR,path_dyamond_dir_utc_time +'.'+var+'.nc')

                X = open_xarray_rolling_lon(file_path = file_path, 
                            lat_bary = lat, 
                            lon_bary = long, 
                            lat_delta = lat_delta_degrees, 
                            lon_delta = lon_delta_degrees, 
                            output_shape = (lat_delta_pixels, lon_delta_pixels))
            
            if X.shape != (lat_delta_pixels, lon_delta_pixels):
                raise ValueError(f'X.shape != (lat_delta_pixels, lon_delta_pixels), actual size : {X.shape}')
            
            X_data [i,j] = X.astype(np.float32)
            
    return X_data


def get_z_data(dict_specs_mcs):
    z_array =  np.array([dict_specs_mcs['label_toocan_mcs'],
                    dict_specs_mcs['duration_mcs'],
                    np.mean(dict_specs_mcs['average_velocity']).astype(float),
                    dict_specs_mcs['classif_JIRAK'],
                    dict_specs_mcs['dist'],
                    dict_specs_mcs['max_extend'],
                    np.mean(dict_specs_mcs['lat_list_mcs']).astype(float),
                    np.mean(dict_specs_mcs['lon_list_mcs']).astype(float)])
    
    return z_array

def get_y_data(object_mcs, start_time=None, end_time=None):
    ### DEPRECIATED -> not used !!
    pass
    """if start_time is not None:
        assert end_time is not None
        duration_mcs = object_mcs.duration
        
    idx_max_area_mcs = np.argmax(object_mcs.clusters.surfkm2_172Wm2)
    max_area_mcs = object_mcs.clusters.surfkm2_172Wm2[idx_max_area_mcs]
    
    t_max_area_mcs = object_mcs.clusters.LC_UTC_time[idx_max_area_mcs]
    t_start_mcs = object_mcs.clusters.LC_UTC_time[0]
    t_end_mcs = object_mcs.clusters.LC_UTC_time[-1]

    delta_t_max_extension_mcs = compute_delta_time(t_start_mcs, t_max_area_mcs)
    
    assert duration_mcs>DeepFate.config.MCS_SPECS_RANGE['duration_hour'][0]
    assert max_area_mcs>DeepFate.config.MCS_SPECS_RANGE['max_area'][0]
    
    return np.array([duration_mcs, delta_t_max_extension_mcs, max_area_mcs]).astype(np.float32)"""

def precompute_single_mcs(mcs_object, start_time, end_time):

    df_relation_table = pd.read_csv(DeepFate.config.PATH_RELATION_TABLE)
    dict_specs_mcs = get_specs_mcs(mcs_object, start_time, end_time) 
    
    #if not (end_time >= start_time+5):#it is hard coded!!
        #print(start_time, end_time)
    
    assert mcs_object.DCS_number == dict_specs_mcs['label_toocan_mcs']
    
    #try:
    X = get_X_data(label_mcs = dict_specs_mcs['label_toocan_mcs'],
                   utc_list = dict_specs_mcs['utc_list_mcs'],
                   lat_list = dict_specs_mcs['lat_list_mcs'],
                   long_list = dict_specs_mcs['lon_list_mcs'],
                   df_relation_table = df_relation_table)

    y  = np.array([dict_specs_mcs['duration_mcs'], dict_specs_mcs['max_extend'],dict_specs_mcs['delta_time_maximal_extension']])
    # oui mcs_object.duration donne plus la duration !! 
    z  = get_z_data(dict_specs_mcs)
    
    assert dict_specs_mcs['duration_mcs'] == (end_time - start_time + 1)/2 
        
        
    """except Exception as e:
        print(' -- -- -- ERROR -- -- -- ')
        print(e)
        print('LABEL MCS', mcs_object.label)
        
        lat_delta_pixels = DeepFate.config.SPACE_WINDOW['lat_delta_pixels']
        lon_delta_pixels = DeepFate.config.SPACE_WINDOW['lon_delta_pixels']
        nb_timesteps = DeepFate.config.NB_TIMESTEPS
        input_variables = DeepFate.config.INPUT_VARIABLES
        X = np.zeros((nb_timesteps, len(input_variables), lon_delta_pixels, lat_delta_pixels))
        y = np.zeros((3,))
        z = np.zeros((7,))"""
    
    return X, y, z
    
            
################### MAIN SCRIPT TO PRECOMPUTE DATASET

def precompute_all_mcs(start_index, stop_index, path_output_h5_file, parallelize:bool=True, nb_workers:int=64, nb_batchs:int=100):
    
    nb_batchs = int((stop_index-start_index)/256)+1
    
    ## FILTER VALID MCS ##
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
    
    #/!! for paralelization purpose
    list_valid_mcs_2 = list_valid_mcs_2[start_index:stop_index]
    list_start_times = list_start_times[start_index:stop_index]
    list_end_times = list_end_times[start_index:stop_index]
    
    nb_timesteps =  DeepFate.config.NB_TIMESTEPS
    #df_relation_table = pd.read_csv(DeepFate.config.PATH_RELATION_TABLE)
    input_variables = INPUT_VARIABLES
    space_window_pixels = (DeepFate.config.SPACE_WINDOW['lon_delta_pixels'],DeepFate.config.SPACE_WINDOW['lat_delta_pixels']) 
    
    ## INSTANTIATE H5 FILES
    X_full_shape = (len(list_valid_mcs_2), nb_timesteps, len(input_variables), space_window_pixels[0], space_window_pixels[1])
    y_full_shape = (len(list_valid_mcs_2), 3) 
    z_full_shape = (len(list_valid_mcs_2), 8)

    with h5py.File(path_output_h5_file, 'w') as hf:
        X_dataset = hf.create_dataset('X', X_full_shape, chunks=((1,) +  tuple(X_full_shape[1:])))
        y_dataset = hf.create_dataset('y', y_full_shape)
        z_dataset = hf.create_dataset('z', z_full_shape)
        
        if not parallelize :
            #(mcs_object, start_time, end_time) 
            for i, (object_mcs, start_time, end_time) in tqdm.tqdm(enumerate(zip(list_valid_mcs_2, list_start_times, list_end_times)), total = len(list_valid_mcs_2)):
                
                X_i, y_i, z_i = precompute_single_mcs(object_mcs, start_time, end_time)
                X_dataset[i] = X_i
                y_dataset[i] = y_i
                z_dataset[i] = z_i
                
        else:
            
            splitted_mcs_list = np.array_split(list_valid_mcs_2, nb_batchs)
            splitted_start_list = np.array_split(list_start_times, nb_batchs)
            splitted_end_list = np.array_split(list_end_times, nb_batchs)
            splitted_idx = np.array_split(np.arange(len(list_valid_mcs_2)), nb_batchs)

            for j, (batch_mcs, batch_start_times, batch_end_times, batch_idx) in enumerate(zip(splitted_mcs_list,splitted_start_list,splitted_end_list, splitted_idx)):
                
                input_precomputing = [(a,b,c) for (a,b,c) in zip(batch_mcs, batch_start_times, batch_end_times)]
                
                print(f'BATCH {j+1} / {nb_batchs}')
            
                with Pool(nb_workers) as p:
                    outputs = p.starmap(precompute_single_mcs, tqdm.tqdm(input_precomputing))

                    X_dataset[batch_idx] = np.array([out[0] for out in outputs]).astype(np.float32)
                    y_dataset[batch_idx] = np.array([out[1] for out in outputs]).astype(np.float16)
                    z_dataset[batch_idx] = np.array([out[2] for out in outputs]).astype(np.float32)

                
                
                
            
            
