from scipy.ndimage import uniform_filter
import numpy as np
import xarray as xr
import os

from utils_path import get_path_dict_from_timesteps

from config_const import ROOT_DICT_PATHS, FOLDER_PATH, MERGED_TABLE, SPECIFIC_HEAT_CAPACITY, LATENT_HEAT_VAPORIZATION, GRAVITY, WINDOW_SIZE_X, WINDOW_SIZE_Y, BOUNDARY_LAYER_HEIGHT, FMSE_CAP, LOW_MID_TROPOSPHERE, HIGH_MID_TROPOSPHERE

"""
def get_fmse_detrended(path_tabs, path_qv, cp=SPECIFIC_HEAT_CAPACITY, Lv=LATENT_HEAT_VAPORIZATION, gravity=GRAVITY,
                       window_size_x=WINDOW_SIZE_X, window_size_y=WINDOW_SIZE_Y, zbl=BOUNDARY_LAYER_HEIGHT):

    #Compute the detrended FMSE by subtracting local mean filtered FMSE values.

    T = xr.open_dataset(path_tabs).TABS.sel(z=slice(0, zbl), lat=slice(-35, 35))[0].chunk({'lat': 512, 'lon': 1024})
    QV = xr.open_dataset(path_qv).QV.sel(z=slice(0, zbl), lat=slice(-35, 35))[0].chunk({'lat': 512, 'lon': 1024})

    Z = T.z
    Z_broadcasted = Z.expand_dims({'lat': T.lat, 'lon': T.lon}).transpose('z', 'lat', 'lon')

    # FMSE computation (QV in g/kg -> kg/kg conversion)
    FMSE = cp * T.values + Lv * QV.values / 1000 + gravity * Z_broadcasted.values

    # Apply local mean filter across spatial dimensions
    local_mean = uniform_filter(FMSE, size=(1, window_size_x, window_size_y), mode='constant')
    FMSE_detrended = FMSE - local_mean

    return FMSE_detrended, Z_broadcasted
"""

def get_fmse_int_bl(path_tabs, path_qv, zbl=BOUNDARY_LAYER_HEIGHT, cp=SPECIFIC_HEAT_CAPACITY, Lv=LATENT_HEAT_VAPORIZATION, gravity=GRAVITY):
    """
    Compute the detrended FMSE by subtracting local mean filtered FMSE values.
    """
    T = xr.open_dataset(path_tabs).TABS.sel(z=slice(0, zbl), lat=slice(-35, 35))[0].chunk({'lat': 512, 'lon': 1024})
    QV = xr.open_dataset(path_qv).QV.sel(z=slice(0, zbl), lat=slice(-35, 35))[0].chunk({'lat': 512, 'lon': 1024})

    Z = T.z
    Z_broadcasted = Z.expand_dims({'lat': T.lat, 'lon': T.lon}).transpose('z', 'lat', 'lon')

    # FMSE computation (QV in g/kg -> kg/kg conversion)
    FMSE = cp * T.values + Lv * QV.values / 1000 + gravity * Z_broadcasted.values

    # Apply local mean filter across spatial dimensions

    dz = np.gradient(Z_broadcasted.values, axis=0)
    fmse_integral_bl = np.sum(FMSE * dz, axis=0) / zbl  # Sum over the z dimension

    return fmse_integral_bl


def get_fmse_int(path_tabs, path_qv, z_low, z_high, cp=SPECIFIC_HEAT_CAPACITY, Lv=LATENT_HEAT_VAPORIZATION, gravity=GRAVITY):
    """
    Compute the detrended FMSE by subtracting local mean filtered FMSE values.
    """
    T = xr.open_dataset(path_tabs).TABS.sel(z=slice(z_low, z_high), lat=slice(-35, 35))[0].chunk({'lat': 512, 'lon': 1024})
    QV = xr.open_dataset(path_qv).QV.sel(z=slice(z_low, z_high), lat=slice(-35, 35))[0].chunk({'lat': 512, 'lon': 1024})

    Z = T.z
    Z_broadcasted = Z.expand_dims({'lat': T.lat, 'lon': T.lon}).transpose('z', 'lat', 'lon')

    # FMSE computation (QV in g/kg -> kg/kg conversion)
    FMSE = cp * T.values + Lv * QV.values / 1000 + gravity * Z_broadcasted.values

    

    # Apply local mean filter across spatial dimensions

    dz = np.gradient(Z_broadcasted.values, axis=0)
    fmse_integral = np.sum(FMSE * dz, axis=0) / (z_high-z_low)  # Sum over the z dimension

    

    return fmse_integral

def get_diff_fmse_mid_bl(path_tabs, path_qv, z_bl, z_low, z_high, cp=SPECIFIC_HEAT_CAPACITY, Lv=LATENT_HEAT_VAPORIZATION, gravity=GRAVITY):
    fmse_integral_mid = get_fmse_int(path_tabs, path_qv, z_low, z_high) 
    fmse_integral_bl = get_fmse_int_bl(path_tabs, path_qv, z_bl)

    diff = fmse_integral_mid - fmse_integral_bl


    return fmse_integral_mid, fmse_integral_bl, diff

def get_decomposition_detrended_fmse(FMSE_detrended):
    """
    Decompose detrended FMSE into positive and negative components.
    """
    positive_FMSE = np.maximum(FMSE_detrended, 0)
    negative_FMSE = np.minimum(FMSE_detrended, 0)
    
    print(np.mean(negative_FMSE * 1000))  # Conversion to consistent units for logging
    return positive_FMSE, negative_FMSE

def integrate_bl(Z_broadcasted, positive_FMSE, negative_FMSE, zbl=BOUNDARY_LAYER_HEIGHT):
    """
    Integrate the positive and negative FMSE components over the boundary layer depth.
    """
    dz = np.gradient(Z_broadcasted.values, axis=0)
    positive_integral = np.sum(positive_FMSE * dz, axis=0) / zbl  # Sum over the z dimension
    negative_integral = np.sum(negative_FMSE * dz, axis=0) / zbl  # Sum over the z dimension

    return positive_integral, negative_integral

def get_fmse_decomposition_from_vars(path_tabs, path_qv, 
                                     cp=SPECIFIC_HEAT_CAPACITY, Lv=LATENT_HEAT_VAPORIZATION, 
                                     gravity=GRAVITY, window_size_x=WINDOW_SIZE_X, window_size_y=WINDOW_SIZE_Y,
                                     zbl=BOUNDARY_LAYER_HEIGHT, cap_fmse=FMSE_CAP):
    """
    Full process to compute detrended FMSE, decompose it, and integrate over the boundary layer.
    """
    FMSE_detrended, Z_broadcasted = get_fmse_detrended(path_tabs=path_tabs, path_qv=path_qv, 
                                                       cp=cp, Lv=Lv, gravity=gravity, 
                                                       window_size_x=window_size_x, window_size_y=window_size_y,
                                                       zbl=zbl)
    
    positive_FMSE, negative_FMSE = get_decomposition_detrended_fmse(FMSE_detrended)
    positive_integral, negative_integral = integrate_bl(Z_broadcasted, positive_FMSE, negative_FMSE, zbl=zbl)

    # Apply caps to the integrals
    positive_integral = np.clip(positive_integral, None, cap_fmse[1])
    negative_integral = np.clip(negative_integral, cap_fmse[0], None)

    return FMSE_detrended, Z_broadcasted, positive_integral, negative_integral

def save_fmse_to_netcdf(FMSE_detrended, positive_integral, negative_integral, Z_broadcasted, path_tabs, 
                        folder_path=FOLDER_PATH):
    """
    Save FMSE detrended and integrated positive/negative FMSE to NetCDF files.
    """
    lat, lon, z = Z_broadcasted.lat, Z_broadcasted.lon, Z_broadcasted.z

    FMSE_detrended_da = xr.DataArray(FMSE_detrended, coords={'z': z, 'lat': lat, 'lon': lon}, dims=['z', 'lat', 'lon'], 
                                     name='FMSE_detrended')
    positive_integral_da = xr.DataArray(positive_integral, coords={'lat': lat, 'lon': lon}, dims=['lat', 'lon'], 
                                        name='positive_integral')
    negative_integral_da = xr.DataArray(negative_integral, coords={'lat': lat, 'lon': lon}, dims=['lat', 'lon'], 
                                        name='negative_integral')

    _, filename = os.path.split(path_tabs)
    base_filename = filename.replace('TABS.nc', 'FMSE_BL_DETRENDED.nc')

    FMSE_detrended_da.to_netcdf(os.path.join(folder_path, base_filename))
    positive_integral_da.to_netcdf(os.path.join(folder_path, base_filename.replace('FMSE_BL', 'INT_POS_FMSE_BL')))
    negative_integral_da.to_netcdf(os.path.join(folder_path, base_filename.replace('FMSE_BL', 'INT_NEG_FMSE_BL')))

def get_and_save_diff_fmse_to_netcdf(path_tabs, path_qv, z_bl, z_low, z_high, folder_path=FOLDER_PATH, cp=SPECIFIC_HEAT_CAPACITY, Lv=LATENT_HEAT_VAPORIZATION, gravity=GRAVITY):

    fmse_integral_mid, fmse_integral_bl, diff_fmse = get_diff_fmse_mid_bl(path_tabs, path_qv, z_bl, z_low, z_high, cp=cp, Lv=Lv, gravity=gravity)

    TABS = xr.open_dataset(path_tabs).TABS.sel(z=slice(z_low, z_high), lat=slice(-35, 35))[0].chunk({'lat': 512, 'lon': 1024})
    lat, lon = TABS.lat, TABS.lon

    diff_fmse_da = xr.DataArray(diff_fmse, coords={'lat': lat, 'lon': lon}, dims=['lat', 'lon'], 
                                        name='diff_fmse_mid_bl')
    
    fmse_integral_mid_da = xr.DataArray(fmse_integral_mid, coords={'lat': lat, 'lon': lon}, dims=['lat', 'lon'], 
                                        name='int_fmse_mid')
    
    fmse_integral_bl_da = xr.DataArray(fmse_integral_bl, coords={'lat': lat, 'lon': lon}, dims=['lat', 'lon'], 
                                        name='int_fmse_bl')

    _, filename = os.path.split(path_tabs)
    base_filename = filename.replace('TABS.nc', 'DIFF_FMSE_MID_BL.nc')
    base_filename2 = filename.replace('TABS.nc', 'INT_FMSE_MID.nc')
    base_filename3 = filename.replace('TABS.nc', 'INT_FMSE_BL.nc')

    diff_fmse_da.to_netcdf(os.path.join(folder_path, base_filename))
    fmse_integral_mid_da.to_netcdf(os.path.join(folder_path, base_filename2))
    fmse_integral_bl_da.to_netcdf(os.path.join(folder_path, base_filename3))


def get_and_save_fmse_bl(path_tabs, path_qv, folder_path=FOLDER_PATH, 
                         cp=SPECIFIC_HEAT_CAPACITY, Lv=LATENT_HEAT_VAPORIZATION, 
                         gravity=GRAVITY, window_size_x=WINDOW_SIZE_X, window_size_y=WINDOW_SIZE_Y,
                         zbl=BOUNDARY_LAYER_HEIGHT, cap_fmse=FMSE_CAP):
    """
    Process the FMSE data and save the outputs to NetCDF files.
    """
    FMSE_detrended, Z_broadcasted, positive_integral, negative_integral = get_fmse_decomposition_from_vars(
        path_tabs=path_tabs, path_qv=path_qv, cp=cp, Lv=Lv, gravity=gravity, 
        window_size_x=window_size_x, window_size_y=window_size_y, zbl=zbl, cap_fmse=cap_fmse)
    
    save_fmse_to_netcdf(FMSE_detrended, positive_integral, negative_integral, Z_broadcasted, path_tabs, folder_path)

def get_fmse_index(i, merged_df, root_dict, folder_path='/work/bb1153/b381993/computed_dyamond_2d'):
    """
    Retrieve FMSE data for a specific timestep and save the processed data to NetCDF files.
    """

    dict_all_path = get_path_dict_from_timesteps(i, merged_df, root_dict)
    path_tabs = os.path.join(root_dict['root_3d_files'], f"{dict_all_path['path_dyamond_3d']}_TABS.nc")
    path_qv = os.path.join(root_dict['root_3d_files'], f"{dict_all_path['path_dyamond_3d']}_QV.nc")

    get_and_save_fmse_bl(path_tabs=path_tabs, path_qv=path_qv, folder_path=folder_path)


def get_fmse_index_new(i, merged_df, root_dict, z_bl=BOUNDARY_LAYER_HEIGHT, z_low=LOW_MID_TROPOSPHERE, z_high=HIGH_MID_TROPOSPHERE, folder_path='/work/bb1153/b381993/computed_dyamond_2d'):
    """
    Retrieve FMSE data for a specific timestep and save the processed data to NetCDF files.
    """

    dict_all_path = get_path_dict_from_timesteps(i, merged_df, root_dict)

    print(dict_all_path)
    path_tabs = os.path.join(root_dict['root_3d_files'], f"{dict_all_path['path_dyamond_3d']}_TABS.nc")
    path_qv = os.path.join(root_dict['root_3d_files'], f"{dict_all_path['path_dyamond_3d']}_QV.nc")

    get_and_save_diff_fmse_to_netcdf(path_tabs=path_tabs, path_qv=path_qv, z_bl=z_bl, z_low=z_low, z_high=z_high, folder_path=folder_path)
