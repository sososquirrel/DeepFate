
import numpy as np
import copy
from scipy import signal
from timezonefinder import TimezoneFinder
import pytz
from datetime import datetime
import pandas as pd
from functools import wraps
import DeepFate
from DeepFate.config import NB_TIMESTEPS, INPUT_VARIABLES



NB_TIMESTEPS = NB_TIMESTEPS
NB_VARIABLES = len(INPUT_VARIABLES)

t_arr = np.linspace(-10, 10, NB_TIMESTEPS)
BUMP = np.exp(-0.02*t_arr**2)
BUMP /= np.trapz(BUMP) # normalize the integral to 1

import time

def timeit(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        print(f'Function {func.__name__} Took {total_time:.4f} seconds')
        return result
    return timeit_wrapper

#@timeit
def create_masks(X_images, z_specs):
    X_mask = X_images[:,[0],:,:]
    mcs_label = z_specs[0]
    X_mask_mcs = copy.deepcopy(X_mask)
    X_mask_mcs[X_mask_mcs!=mcs_label] = 0
    X_mask_mcs[X_mask_mcs==mcs_label] = 1
    X_mask_neighbours = copy.deepcopy(X_mask)
    X_mask_neighbours[X_mask_neighbours>0.5] = 1 
    X_mask_neighbours = X_mask_neighbours - X_mask_mcs
    X_anti_mask_mcs = 1 - X_mask_mcs
    return X_mask_mcs, X_mask_neighbours, X_anti_mask_mcs

#@timeit
def get_migration_distance_from_birth(mcs_object, start_index):
    """Migration Distance (Southern Ocean mesocyclones from manually tracked satellite mosaics)"""
    CONVERT_DEG_KM = 111.1
    x_dist_km = CONVERT_DEG_KM*(np.array(mcs_object.clusters.LC_lat[start_index:start_index+NB_TIMESTEPS]) -  mcs_object.clusters.LC_lat[start_index])
    y_dist_km = CONVERT_DEG_KM*(np.array(mcs_object.clusters.LC_lon[start_index:start_index+NB_TIMESTEPS]) -  mcs_object.clusters.LC_lon[start_index])
    dists = np.sqrt(x_dist_km**2 + y_dist_km**2)
    names = np.array([f'migration_dist_time_{i}' for i in range(NB_TIMESTEPS)])
    assert dists.shape == names.shape
    return names, dists

#@timeit
def get_absolute_position_from_birth(mcs_object, start_index):
    lat = mcs_object.clusters.LC_lat[start_index:start_index+NB_TIMESTEPS]
    lon = mcs_object.clusters.LC_lon[start_index:start_index+NB_TIMESTEPS]
    def convert(lon):
        if lon>180:
            return lon-180
        return lon
    lon = [convert(l) for l in lon]
    out = np.array(lat + lon)
    names = np.array([f'lat_time_{i}' for i in range(NB_TIMESTEPS)] + [f'lon_time_{i}' for i in range(NB_TIMESTEPS)])
    assert names.shape == out.shape
    return names, out

def get_mcs_area(X_mask_mcs):
    areas = np.sum(X_mask_mcs[:,0], axis=(-1, -2)).flatten()
    names = np.array([f'mcs_area_time_{i}' for i in range(NB_TIMESTEPS)])
    assert areas.shape == names.shape
    return names, areas

def get_all_mcs_area(X_mask_mcs, X_mask_neighbours):
    areas_mcs = np.sum(X_mask_mcs[:,0], axis=(-1, -2)).flatten()
    areas_all_mcs = np.sum(X_mask_neighbours[:,0], axis=(-1, -2)).flatten()
    sum_areas = areas_mcs + areas_all_mcs
    names = np.array([f'all_mcs_area_time_{i}' for i in range(NB_TIMESTEPS)])
    assert sum_areas.shape == names.shape
    return names, sum_areas

#@timeit
def _utc_to_local(utc_time_str, lat, lon):
    tf = TimezoneFinder()
    timezone_str = tf.certain_timezone_at(lat=lat, lng=lon)
    timezone = pytz.timezone(timezone_str)
    str_time = datetime.fromtimestamp(utc_time_str).strftime("%Y-%m-%dT%H:%M:%S")
    utc_time = datetime.strptime(str_time, "%Y-%m-%dT%H:%M:%S")
    local_time = utc_time.replace(tzinfo=pytz.utc).astimezone(timezone)
    return local_time.hour

#@timeit
def get_local_hour(mcs_object, start_index):
    #TODO circularity in hours to add ! 
    positions = get_absolute_position_from_birth(mcs_object, start_index)[1]
    utimes = mcs_object.clusters.LC_UTC_time[start_index:start_index+NB_TIMESTEPS]
    #local_hours = np.array([_utc_to_local(float(t), lat =float(lat), lon = float(lon)) for (t,lat,lon) in zip(utimes, positions[:NB_TIMESTEPS], positions[NB_TIMESTEPS:])])
    names = np.array([f'local_hour_time_{i}' for i in range(NB_TIMESTEPS)])
    assert local_hours.shape == names.shape
    return names, local_hours

#@timeit
def get_propagation_velocity(mcs_object, start_index):
    dist = get_migration_distance_from_birth(mcs_object, start_index)[1]
    v_propag = np.gradient(dist, 0.5)
    names = np.array([f'propagation_velocity_time_{i}' for i in range(NB_TIMESTEPS)])
    assert v_propag.shape == names.shape
    return names, v_propag

#@timeit
def get_gradient_area(X_mask_mcs):
    area_masks = np.sum(X_mask_mcs[:,0], axis=(-1,-2))
    gradient_area = np.gradient(area_masks, 0.5)
    names = np.array([f'gradient_area_time_{i}' for i in range(NB_TIMESTEPS)])
    assert gradient_area.shape == names.shape
    return names, gradient_area

#@timeit
def get_mean_everywhere(X_images):
    means = np.mean(X_images[:,1:], axis=(-1, -2)).flatten()
    names = np.array([f'mean_everywhere_var_{j}_time_{i}' for i in range(NB_TIMESTEPS) for j in range(1, NB_VARIABLES)])
    assert means.shape == names.shape
    return names, means

#@timeit
def get_std_everywhere(X_images):
    means = np.std(X_images[:,1:], axis=(-1, -2)).flatten()
    names = np.array([f'std_everywhere_var_{j}_time_{i}' for i in range(NB_TIMESTEPS) for j in range(1, NB_VARIABLES)])
    assert means.shape == names.shape
    return names, means
 
#@timeit
def get_mean_under_cloud(X_images, X_mask_mcs):
    X_images_masked = X_mask_mcs * X_images[:,1:]
    means = np.mean(X_images_masked, axis=(-1, -2)).flatten()
    names = np.array([f'mean_under_cloud_var_{j}_time_{i}' for i in range(NB_TIMESTEPS) for j in range(1, NB_VARIABLES)])
    assert means.shape == names.shape
    return names, means

#@timeit
def get_std_under_cloud(X_images, X_mask_mcs):
    X_images_masked = X_mask_mcs * X_images[:,1:]
    means = np.std(X_images_masked, axis=(-1, -2)).flatten()
    names = np.array([f'std_under_cloud_var_{j}_time_{i}' for i in range(NB_TIMESTEPS) for j in range(1, NB_VARIABLES)])
    assert means.shape == names.shape
    return names, means
 
#@timeit
def get_mean_everywhere_except_cloud(X_images, X_anti_mask_mcs):
    X_images_masked = X_anti_mask_mcs * X_images[:,1:]
    means = np.mean(X_images_masked, axis=(-1, -2)).flatten()
    names = np.array([f'mean_everywhere_except_cloud_var_{j}_time_{i}' for i in range(NB_TIMESTEPS) for j in range(1, NB_VARIABLES)])
    assert means.shape == names.shape
    return names, means
#@timeit
def get_mean_under_all_mcs_neighbours(X_images,X_mask_neighbours):
    X_images_masked = X_mask_neighbours * X_images[:,1:,:,:]
    means = np.mean(X_images_masked, axis=(-1, -2)).flatten()
    names = np.array([f'mean_under_all_mcs_neighbours_var_{j}_time_{i}' for i in range(NB_TIMESTEPS) for j in range(1,NB_VARIABLES)])
    assert means.shape == names.shape
    return names, means
#@timeit
def _get_divergence_omega(X_images):
    U = X_images[:,-6, :,:]
    V = X_images[:,-5, :,:]
    dx_U = np.gradient(U, axis=-2)
    dy_V = np.gradient(V, axis=-1)
    return np.array((dx_U + dy_V))

#@timeit
def get_div_multiple_means(X_images, X_mask_mcs, X_mask_neighbours, X_anti_mask_mcs):
    div = np.array(_get_divergence_omega(X_images))
    div_masked = X_mask_mcs[:,0] * div
    div_masked_neighbours = X_mask_neighbours[:,0] * div
    div_masked_anti_mcs = X_anti_mask_mcs[:,0] * div 
    out = np.array([np.mean(div, axis=(-1, -2)), 
                     np.mean(div_masked, axis=(-1, -2)), 
                     np.mean(div_masked_neighbours, axis=(-1, -2)),  
                     np.mean(div_masked_anti_mcs, axis=(-1, -2))]).flatten()
    names = np.array([f'div_full_image_time_{i}' for i in range(NB_TIMESTEPS)] + [f'div_mask_mcs_time_{i}' for i in range(NB_TIMESTEPS)] + [f'div_mask_neighbours_time_{i}' for i in range(NB_TIMESTEPS)] + [f'div_anti_mask_time_{i}' for i in range(NB_TIMESTEPS)])
    assert out.shape == names.shape
    return names, out

#@timeit
def get_mean_divergence_omega(X_images):
    U = X_images[:,-6, :,:]
    V = X_images[:,-5, :,:]
    dx_U = np.gradient(U, axis=-2)
    dy_V = np.gradient(V, axis=-1)
    out = np.mean(dx_U + dy_V, axis=(-1, -2)).flatten()
    names = np.array([f'mean_divergence_omega_time_{i}' for i in range(NB_TIMESTEPS)])
    assert out.shape == names.shape
    return names, out

#@timeit
def number_of_neighbour(X_images):
    out = np.array([len(np.unique(X_images[t,0,:,:])-2) for t in range(NB_TIMESTEPS)]) #remove env and the system so -2
    names = np.array([f'nb_neighbours_time_{i}' for i in range(NB_TIMESTEPS)])
    assert out.shape == names.shape
    return names, out

#@timeit
def number_of_active_neighbour(X_images, array_index_label):
    outs = []
    for t in range(NB_TIMESTEPS):
        label_list=[]
        for label in np.unique(X_images[t,0,:,:]):
            if label in array_index_label:
                label_list.append(label)
        outs.append(len(label_list)-1) #remove the system so -1
    outs = np.array(outs)
    names = np.array([f'number_of_active_neighbour_time_{i}' for i in range(NB_TIMESTEPS)])
    assert outs.shape == names.shape
    return names, outs

#@timeit
def get_mean_position_neighbour(X_images, specs):
    nx, ny = X_images[0,0,:,:].shape
    PIX_TO_KM = 4 #resolution
    mcs_label=specs[0]
    list_dist_to_neighbour=[]
    for t in range(NB_TIMESTEPS):
        list_distance_at_each_timsteps = []
        for label in np.unique(X_images[t,0,:,:]):
            if (label==mcs_label) or (label==0):
                None
            else:
                x_bary_neighbour, y_bary_neighbour = np.mean(np.where(X_images[t,0,:,:]==label)[0]),  np.mean(np.where(X_images[t,0,:,:]==label)[1]) #
                dist_neighbour_label = PIX_TO_KM*np.sqrt((x_bary_neighbour-nx/2)**2 + (y_bary_neighbour-ny/2)**2)
                list_distance_at_each_timsteps.append(dist_neighbour_label)
        mean = np.mean(np.array(list_distance_at_each_timsteps)) if len(list_distance_at_each_timsteps)>0 else (nx+1)*PIX_TO_KM
        list_dist_to_neighbour.append(mean)
        
    out = np.array(list_dist_to_neighbour)
    names = np.array([f'mean_position_neighbour_time_{i}' for i in range(NB_TIMESTEPS)])
    assert out.shape == names.shape
    return names, out

#@timeit
def get_min_position_neighbour(X_images, specs):
    nx, ny = X_images[0,0,:,:].shape
    PIX_TO_KM = 4 #resolution
    mcs_label=specs[0]
    list_dist_to_neighbour=[]
    for t in range(NB_TIMESTEPS):
        list_distance_at_each_timsteps = []
        for label in np.unique(X_images[t,0,:,:]):
            if (label==mcs_label) or (label==0):
                None
            else:
                x_bary_neighbour, y_bary_neighbour = np.mean(np.where(X_images[t,0,:,:]==label)[0]),  np.mean(np.where(X_images[t,0,:,:]==label)[1]) #
                dist_neighbour_label = PIX_TO_KM*np.sqrt((x_bary_neighbour-nx/2)**2 + (y_bary_neighbour-ny/2)**2)
                list_distance_at_each_timsteps.append(dist_neighbour_label)
        mini = np.min(np.array(list_distance_at_each_timsteps)) if len(list_distance_at_each_timsteps)>0 else nx*PIX_TO_KM
        list_dist_to_neighbour.append(mini)
    out = np.array(list_dist_to_neighbour)
    names = np.array([f'min_position_neighbour_time_{i}' for i in range(NB_TIMESTEPS)])
    assert out.shape == names.shape
    return names, out
 
#@timeit
def get_average_age_neighbour(X_images, specs, list_mcs_object, array_index_label, start_list):
    mcs_label=specs[0]
    list_ages_neighbour=[]
    mcs_index = int(np.where(array_index_label == specs[0])[0][0])
    mcs_object = list_mcs_object[mcs_index]
    start = start_list[mcs_index]
    list_utime_mcs = mcs_object.clusters.LC_UTC_time[start:start+NB_TIMESTEPS]
    for t in range(NB_TIMESTEPS):
        list_ages_at_each_timsteps = []
        for label in np.unique(X_images[t,0,:,:]):
            if (label==mcs_label) or (label==0):
                None
            else:
                label=int(label)
                if label in array_index_label:
                    mcs_neighbour_index = int(np.where(array_index_label == label)[0][0])
                    mcs_neighbour_object = list_mcs_object[mcs_neighbour_index]
                    start_neighbour = start_list[mcs_neighbour_index]
                    list_utime_mcs_neighbour = mcs_neighbour_object.clusters.LC_UTC_time[start_neighbour:start_neighbour+NB_TIMESTEPS]
                    list_intersect=np.where(np.array(list_utime_mcs_neighbour) == list_utime_mcs[t])[0]
                    if len(list_intersect)!=0:
                        age_mcs_neighbour = 0.5 * list_intersect[0] # in hour
                        list_ages_at_each_timsteps.append(age_mcs_neighbour)
                    else:
                        None
                else:
                    None
                    
            mean = np.min(np.mean(np.array(list_ages_at_each_timsteps))) if len(list_ages_at_each_timsteps)>0 else 0
        list_ages_neighbour.append(mean)
    
    out =  np.array(list_ages_neighbour)
    names = np.array([f'average_age_neighbour_time_{i}' for i in range(NB_TIMESTEPS)])
    assert out.shape == names.shape
    return names, out

#@timeit
def get_max_age_neighbour(X_images, specs, list_mcs_object, array_index_label, start_list):
    mcs_label=specs[0]
    list_ages_neighbour=[]
    mcs_index = int(np.where(array_index_label == specs[0])[0][0])
    mcs_object = list_mcs_object[mcs_index]
    start = start_list[mcs_index]
    list_utime_mcs = mcs_object.clusters.LC_UTC_time[start:start+NB_TIMESTEPS]
    for t in range(NB_TIMESTEPS):
        list_ages_at_each_timsteps = []
        for label in np.unique(X_images[t,0,:,:]):
            if (label==mcs_label) or (label==0):
                None
            else:
                label=int(label)
                if label in array_index_label:
                    mcs_neighbour_index = int(np.where(array_index_label == label)[0][0])
                    mcs_neighbour_object = list_mcs_object[mcs_neighbour_index]
                    start_neighbour = start_list[mcs_neighbour_index]
                    list_utime_mcs_neighbour = mcs_neighbour_object.clusters.LC_UTC_time[start_neighbour:start_neighbour+NB_TIMESTEPS]
                    list_intersect=np.where(np.array(list_utime_mcs_neighbour) == list_utime_mcs[t])[0]
                    if len(list_intersect)!=0:
                        age_mcs_neighbour = 0.5 * list_intersect[0] # in hour
                        list_ages_at_each_timsteps.append(age_mcs_neighbour)
                    else:
                        None
                else:
                    None
                
        if len(list_ages_at_each_timsteps)==0:
            list_ages_neighbour.append(0)
        else:
            list_ages_neighbour.append(np.max(np.array(list_ages_at_each_timsteps)))
    out = np.array(list_ages_neighbour)
    names = np.array([f'max_age_neighbour_time_{i}' for i in range(NB_TIMESTEPS)])
    assert out.shape == names.shape
    return names, out

#@timeit
def long_lived_neighbour_detected(X_images, specs, list_mcs_object, array_index_label, start_list):
    list_max_ages = np.array(get_max_age_neighbour(X_images=X_images, 
                                                   specs=specs, 
                                                   list_mcs_object=list_mcs_object, 
                                                   array_index_label=array_index_label, 
                                                   start_list=start_list)[1])
    list_max_ages[list_max_ages<5]=0
    list_max_ages[list_max_ages>5]=1
    names = np.array([f'long_lived_neighbour_detected_time_{i}' for i in range(NB_TIMESTEPS)])
    assert list_max_ages.shape == names.shape
    return names, list_max_ages

#@timeit
def _gaussian_filter_mcs(msc_mask_snapshot, bump = BUMP):
    kernel = bump[:, np.newaxis] * bump[np.newaxis, :]
    output_filtered = signal.fftconvolve(msc_mask_snapshot, kernel[:, :], mode='same')
    return output_filtered

#@timeit
def mean_interaction_power(X_mask_mcs, X_mask_neighbours):
    # mean of product of gaussian filter with mask all neighbours
    out = []
    for t in range(NB_TIMESTEPS):
        X_mask_gaussian_filter = _gaussian_filter_mcs(X_mask_mcs[t,0], bump = BUMP)
        interaction_power = X_mask_gaussian_filter * X_mask_neighbours
        out.append(np.mean(interaction_power))
    out = np.array(out)
    names = np.array([f' mean_interaction_power_time_{i}' for i in range(NB_TIMESTEPS)])
    assert out.shape == names.shape
    return names, out

#@timeit
def max_interaction_power(X_mask_mcs, X_mask_neighbours):
    # mean of product of gaussian filter with mask all neighbours
    out = []
    for t in range(NB_TIMESTEPS):
        X_mask_gaussian_filter = _gaussian_filter_mcs(X_mask_mcs[t,0], bump = BUMP)
        interaction_power = X_mask_gaussian_filter * X_mask_neighbours
        out.append(np.max(interaction_power))
    out = np.array(out)
    names = np.array([f' max_interaction_power_time_{i}' for i in range(NB_TIMESTEPS)])
    assert out.shape == names.shape
    return names, out

#@timeit
def _get_area_mask(X_mask_mcs,threshold2=0.4):
    idx_contour = np.where(X_mask_mcs>threshold2)
    mask_binary = np.zeros_like(X_mask_mcs)
    mask_binary[idx_contour] = 1
    return mask_binary

#@timeit
def _get_perimeter_mask(X_mask_mcs,threshold1=0.6,threshold2=0.4):
    idx_contour = np.where((X_mask_mcs<threshold1) & (X_mask_mcs>threshold2))
    mask_binary = np.zeros_like(X_mask_mcs)
    mask_binary[idx_contour] = 1
    return mask_binary

#@timeit
def _get_circularity_parameter_mask(X_mask_mcs):
    area = np.sum(_get_area_mask(X_mask_mcs))
    perimeter = np.sum(_get_perimeter_mask(X_mask_mcs))
    p = 4*np.pi*area / (perimeter)**2
    return p

#@timeit
def get_circularity(X_mask_mcs):
    circularity_list=[]
    for t in range(NB_TIMESTEPS):
        X_mask_t = _gaussian_filter_mcs(X_mask_mcs[t,0])
        circularity = _get_circularity_parameter_mask(X_mask_t)
        circularity_list.append(circularity)
    out = np.array(circularity_list)
    names = np.array([f'circularity_time_{i}' for i in range(NB_TIMESTEPS)])
    assert out.shape == names.shape
    return names, out

#@timeit
def average_diameters(X_mask_mcs):
    diameters_list=[]
    for t in range(NB_TIMESTEPS):
        X_mask_t = _gaussian_filter_mcs(X_mask_mcs[t,0])
        diameters = np.sqrt(np.sum(_get_area_mask(X_mask_t,threshold2=0.4))/np.pi)
        diameters_list.append(diameters)
    out =  np.array(diameters_list)
    names = np.array([f'average_diameter_time_{i}' for i in range(NB_TIMESTEPS)])
    assert out.shape == names.shape
    return names, out

#@timeit
def get_eccentricity_132(mcs_object, start_index):
    out = np.array(mcs_object.clusters.LC_ecc_220K[start_index:start_index+NB_TIMESTEPS])
    names = np.array([f'eccentricity_132_time_{i}' for i in range(NB_TIMESTEPS)])
    assert out.shape == names.shape
    return names, out

#@timeit
def get_eccentricity_172(mcs_object, start_index):
    out = np.array(mcs_object.clusters.LC_ecc_235K[start_index:start_index+NB_TIMESTEPS])
    names = np.array([f'eccentricity_172_time_{i}' for i in range(NB_TIMESTEPS)])
    assert out.shape == names.shape
    return names, out

#@timeit
def create_df_from_outputs(names, values):
    df= pd.DataFrame(values[:,None].T, columns = names)
    df.index = [0]
    return df

#@timeit
def get_all_features_single_mcs(X_images, specs, list_valid_mcs_2, label_all, list_start_times):
    mcs_label = specs[0]
    mcs_index = np.where(label_all==mcs_label)[0]
    assert len(mcs_index) == 1
    mcs_index = mcs_index[0]
    mcs_object = list_valid_mcs_2[mcs_index]
    start_index = list_start_times[mcs_index]
    
    X_images = np.nan_to_num(X_images)
    X_mask_mcs, X_mask_neighbours, X_anti_mask_mcs = create_masks(X_images = X_images, z_specs = specs)
    
    ###### NEIGHBORS
    df_nb_of_neighbour = create_df_from_outputs(*number_of_neighbour(X_images=X_images))
    
    
    df_nb_of_active_neighbour = create_df_from_outputs(*number_of_active_neighbour(X_images=X_images, array_index_label=label_all))
    df_mean_position_neighbour = create_df_from_outputs(*get_mean_position_neighbour(X_images=X_images, specs=specs))
    df_min_position_neighbour = create_df_from_outputs(*get_min_position_neighbour(X_images=X_images, specs=specs))
    df_average_age_neighbour =create_df_from_outputs(*get_average_age_neighbour(X_images=X_images, specs=specs, list_mcs_object=list_valid_mcs_2, array_index_label=label_all, start_list=list_start_times))
    df_max_age_neighbour =create_df_from_outputs(*get_max_age_neighbour(X_images=X_images, specs=specs, list_mcs_object=list_valid_mcs_2, array_index_label=label_all, start_list=list_start_times))
    
    df_bool_long_lived_neighbour_detected = create_df_from_outputs(*long_lived_neighbour_detected(X_images=X_images, specs=specs, list_mcs_object=list_valid_mcs_2, array_index_label=label_all, start_list=list_start_times))
    df_mean_interaction_power = create_df_from_outputs(*mean_interaction_power(X_mask_mcs=X_mask_mcs, X_mask_neighbours=X_mask_neighbours))
    df_max_interaction_power = create_df_from_outputs(*max_interaction_power(X_mask_mcs=X_mask_mcs, X_mask_neighbours=X_mask_neighbours))
    

    
    DF_NEIGHBORS = pd.concat([df_nb_of_neighbour, df_nb_of_active_neighbour, df_mean_position_neighbour,
                              df_min_position_neighbour, df_average_age_neighbour, df_max_age_neighbour,
                            df_bool_long_lived_neighbour_detected, df_mean_interaction_power, df_max_interaction_power], axis=1)
    
    #DF_NEIGHBORS = pd.concat([df_nb_of_neighbour, df_nb_of_active_neighbour, df_mean_position_neighbour,
     #                         df_min_position_neighbour, df_mean_interaction_power, df_max_interaction_power], axis=1)
    
    
    ## DISPLACEMENT
    df_migration_distance_from_birth = create_df_from_outputs(*get_migration_distance_from_birth(mcs_object=mcs_object, start_index=start_index))
    df_absolute_position_from_birth =create_df_from_outputs(*get_absolute_position_from_birth(mcs_object=mcs_object, start_index=start_index))
    df_propagation_velocity = create_df_from_outputs(*get_propagation_velocity(mcs_object=mcs_object, start_index=start_index))
    DF_DISPLACEMENT = pd.concat([df_migration_distance_from_birth, df_absolute_position_from_birth], axis=1)
    
    
    
    # TIME 
    #df_local_hour = create_df_from_outputs(*get_local_hour(mcs_object=mcs_object, start_index=start_index))
    #DF_TIME = df_local_hour
    
    # DIVERGENCES
    df_mean_divergence_omega = create_df_from_outputs(*get_mean_divergence_omega(X_images=X_images))
    df_div_multiple_means = create_df_from_outputs(*get_div_multiple_means(X_images= X_images , X_mask_mcs = X_mask_mcs, X_mask_neighbours = X_mask_neighbours, X_anti_mask_mcs=X_anti_mask_mcs))
    DF_DIVS = pd.concat([df_mean_divergence_omega, df_div_multiple_means], axis=1)
    
    # FORM
    df_circularity_param = create_df_from_outputs(*get_circularity(X_mask_mcs=X_mask_mcs))
    df_gradient_area = create_df_from_outputs(*get_gradient_area(X_mask_mcs=X_mask_mcs))
    df_average_diameters = create_df_from_outputs(*average_diameters(X_mask_mcs = X_mask_mcs))
    df_eccentricity_132 = create_df_from_outputs(*get_eccentricity_132(mcs_object=mcs_object, start_index=start_index))
    df_eccentricity_172 = create_df_from_outputs(*get_eccentricity_172(mcs_object=mcs_object, start_index=start_index))
    df_area = create_df_from_outputs(*get_mcs_area(X_mask_mcs=X_mask_mcs))
    df_all_area = create_df_from_outputs(*get_all_mcs_area(X_mask_mcs=X_mask_mcs, X_mask_neighbours = X_mask_neighbours))
    DF_FORM = pd.concat([df_area, df_gradient_area, df_all_area, df_circularity_param, df_average_diameters, df_eccentricity_132, df_eccentricity_172], axis=1)
    
    #VARIABLES MEANS
    df_mean_everywhere = create_df_from_outputs(*get_mean_everywhere(X_images=X_images))
    df_mean_under_cloud = create_df_from_outputs(*get_mean_under_cloud(X_images=X_images, X_mask_mcs = X_mask_mcs))
    df_mean_everywhere_except_cloud = create_df_from_outputs(*get_mean_everywhere_except_cloud(X_images=X_images, X_anti_mask_mcs=X_anti_mask_mcs))
    df_mean_under_all_clouds = create_df_from_outputs(*get_mean_under_all_mcs_neighbours(X_images=X_images, X_mask_neighbours = X_mask_neighbours))
    DF_MEANS = pd.concat([df_mean_everywhere, df_mean_under_cloud, df_mean_everywhere_except_cloud, df_mean_under_all_clouds], axis=1)
    
    ##STDS
    df_std_everywhere = create_df_from_outputs(*get_std_everywhere(X_images=X_images))
    df_std_under_cloud = create_df_from_outputs(*get_std_under_cloud(X_images=X_images, X_mask_mcs=X_mask_mcs))
    DF_STD = pd.concat([df_std_everywhere, df_std_under_cloud], axis=1)

    assert(len(DF_NEIGHBORS)==1)
    assert(len(DF_DISPLACEMENT)==1)
    assert(len(DF_DIVS)==1)
    assert(len(DF_FORM)==1)
    assert(len(DF_MEANS)==1)
    assert(len(DF_STD)==1)
    global_df = pd.concat([DF_NEIGHBORS, DF_DISPLACEMENT, DF_DIVS, DF_FORM, DF_MEANS, DF_STD], axis=1)
    global_df.loc[0, 'label_mcs'] = int(mcs_label)
    global_df.loc[0, 'y_duration'] = specs[1]
    global_df.loc[0, 'y_max_extend'] = specs[5]
    
        
    return global_df



        





