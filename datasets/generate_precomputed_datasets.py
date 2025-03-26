# Standard library imports
import os
import warnings
from multiprocessing import Pool

# Third-party imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr
import h5py
import tqdm

# Custom/Project-specific imports
import DeepFate
from DeepFate.config import PATH_TOOCAN_GLOBAL_FILE, INPUT_VARIABLES
from DeepFate.datasets.load_toocan_dyamond import load_TOOCAN_DYAMOND
from DeepFate.datasets.utils import (
    get_list_valid_mcs, 
    get_validity_lifecycles_start_end, 
    generate_img_seg_file_path_from_utc,
    generate_dyamond_file_path_from_utc, 
    open_xarray_rolling_lon, 
    binary_segmentation_mask_processing, 
    compute_delta_time
)

# Suppress warnings
warnings.simplefilter("ignore")



def get_specs_mcs(object_mcs, start_time, end_time):
    """
    Extracts and calculates various specifications and characteristics of a mesoscale convective system (MCS)
    over a specified time range (`start_time` to `end_time`).

    The function returns a dictionary containing the MCS's properties such as UTC time, latitude, longitude, 
    surface area, velocity, and other related metrics.

    Parameters:
    -----------
    object_mcs : object
        The mesoscale convective system (MCS) object containing the cluster data and MCS-specific attributes.
        This object must have the following attributes:
            - object_mcs.clusters.LC_UTC_time: List of UTC times for each timestep.
            - object_mcs.clusters.LC_lat: List of latitudes for each timestep.
            - object_mcs.clusters.LC_lon: List of longitudes for each timestep.
            - object_mcs.clusters.LC_surfkm2_235K: List of surface area values (in km²).
            - object_mcs.clusters.LC_velocity: List of velocity values for each timestep.
            - object_mcs.INT_classif_JIRAK: Classification data (specific to the MCS).
            - object_mcs.INT_distance: Distance data (specific to the MCS).
            - object_mcs.DCS_number: Label for the MCS in the TOOCAN algorithm.

    start_time : int
        The starting timestep of the MCS to consider for extracting the data.

    end_time : int
        The ending timestep of the MCS to consider for extracting the data.

    Returns:
    --------
    dict_specs_mcs : dict
        A dictionary containing various properties of the MCS, including:
            - 'label_toocan_mcs': MCS label in the TOOCAN algorithm.
            - 'utc_list_mcs': List of UTC times for the specified timesteps.
            - 'lat_list_mcs': List of latitudes for the specified timesteps.
            - 'lon_list_mcs': List of longitudes for the specified timesteps.
            - 'surf_km_list_mcs': List of surface area values (in km²) for the specified timesteps.
            - 'duration_mcs': Duration of the MCS in hours (calculated based on the time window).
            - 'average_velocity': List of velocity values for the specified timesteps.
            - 'classif_JIRAK': Classification information related to the MCS.
            - 'dist': Distance information related to the MCS.
            - 'max_extend': Maximum extension of the MCS based on surface area.
            - 'delta_time_maximal_extension': Time difference between the start time and the timestep
              of maximum extension.

    Raises:
    -------
    ValueError : 
        If the specified time window (`start_time` to `end_time`) is too small for the given number of timesteps,
        or if the maximum surface area occurs outside the specified time window, or if the input data arrays
        are not large enough to support the given number of timesteps.

    Notes:
    ------
    The time window must be large enough to accommodate at least `nb_timesteps` timesteps, which is
    defined by the configuration value `DeepFate.config.NB_TIMESTEPS`. The function assumes that each timestep
    represents 0.5 hours.

    Example:
    --------
    result = get_specs_mcs(object_mcs, start_time=0, end_time=5)
    print(result['utc_list_mcs'])
    """
    
    nb_timesteps = DeepFate.config.NB_TIMESTEPS  # 2, but previously was 10
    
    # Ensure that the time window is large enough to accommodate the required number of timesteps
    if (end_time - start_time) < (nb_timesteps - 1):
        raise ValueError(f"Time window from {start_time} to {end_time} is too small for {nb_timesteps} timesteps.")
    
    # Validate that the object_mcs contains enough data for the specified number of timesteps
    if len(object_mcs.clusters.LC_UTC_time) < nb_timesteps:
        raise ValueError(f"object_mcs.clusters.LC_UTC_time contains fewer than {nb_timesteps} timesteps.")
    
    if len(object_mcs.clusters.LC_lon) < nb_timesteps:
        raise ValueError(f"object_mcs.clusters.LC_lon contains fewer than {nb_timesteps} timesteps.")
    
    # Calculate the duration of the MCS (in hours) based on the time window provided
    duration_mcs = (int(end_time) - int(start_time) + 1) * 0.5  # Assuming each timestep represents 0.5 hours
    
    # Find the timestep corresponding to the maximum surface area
    idx_max = np.argmax(object_mcs.clusters.LC_surfkm2_235K)
    
    # Ensure the maximum surface area occurs within the specified time window
    if not start_time < idx_max < end_time:
        raise ValueError(f"Maximum surface area occurs outside the specified time window: {start_time} -> {end_time}")
        
    # Calculate the maximum extension of the MCS (based on the maximum surface area)
    max_extend = np.sqrt(np.max(object_mcs.clusters.LC_surfkm2_235K))  # Previously 172
    
    # Find the timestep corresponding to the maximum extension and calculate the time difference
    idx_max_extension = np.argmax(object_mcs.clusters.LC_surfkm2_235K)
    delta_time_max_extension = (int(idx_max_extension) - int(start_time)) / 2  # Time difference in hours
    
    # Prepare the dictionary containing the specifications of the MCS
    dict_specs_mcs = {
        'label_toocan_mcs': object_mcs.DCS_number,  # Label of the MCS in the TOOCAN algorithm
        'utc_list_mcs': object_mcs.clusters.LC_UTC_time[start_time:start_time + nb_timesteps],  # UTC time over timesteps
        'lat_list_mcs': object_mcs.clusters.LC_lat[start_time:start_time + nb_timesteps],  # Latitude over timesteps
        'lon_list_mcs': object_mcs.clusters.LC_lon[start_time:start_time + nb_timesteps],  # Longitude over timesteps
        'surf_km_list_mcs': object_mcs.clusters.LC_surfkm2_235K[start_time:start_time + nb_timesteps],  # Surface area over timesteps
        'duration_mcs': duration_mcs,  # Duration of the MCS in hours
        'average_velocity': object_mcs.clusters.LC_velocity[start_time:start_time + nb_timesteps],  # Average velocity over timesteps
        'classif_JIRAK': object_mcs.INT_classif_JIRAK,  # Classification (possibly from a separate system or dataset)
        'dist': object_mcs.INT_distance,  # Distance (possibly to a reference point or system)
        'max_extend': max_extend,  # Maximum extension of the MCS
        'delta_time_maximal_extension': delta_time_max_extension  # Time difference to maximal extension
    }
    
    # Assert that all the lists of MCS properties have the same length
    assert len(set([
        len(dict_specs_mcs['utc_list_mcs']),
        len(dict_specs_mcs['lat_list_mcs']),
        len(dict_specs_mcs['lon_list_mcs']),
        len(dict_specs_mcs['surf_km_list_mcs'])
    ])) == 1, "Mismatched lengths in MCS property lists."
    
    # Return the dictionary containing all MCS specifications
    return dict_specs_mcs


    

def get_X_data(label_mcs, 
               utc_list, 
               lat_list, 
               long_list, 
               df_relation_table, 
               nb_timesteps = DeepFate.config.NB_TIMESTEPS,
               input_variables = DeepFate.config.INPUT_VARIABLES,
               df_merged_2d_3d= None):
    """
    Constructs a dataset (X_data) by gathering various input variables for each timestep from given lists.

    The function processes MCS data for each timestep, handling different variable types:
    - Segmentation images
    - Computed variables (e.g., shear, intensity)
    - Other standard variables
    It checks if the required variables are available and raises errors if necessary inputs are missing.

    Args:
        label_mcs (any): Label for the MCS (not used in the function but possibly relevant to data processing).
        utc_list (list): List of UTC timestamps for each timestep.
        lat_list (list): List of latitude values for each timestep.
        long_list (list): List of longitude values for each timestep.
        df_relation_table (DataFrame): DataFrame containing the relation between time and file paths.
        nb_timesteps (int): Number of timesteps to process (defaults to `DeepFate.config.NB_TIMESTEPS`).
        input_variables (list): List of variables to include in the dataset (defaults to `DeepFate.config.INPUT_VARIABLES`).
        df_merged_2d_3d (DataFrame, optional): DataFrame containing 3D data, required for some computed variables. Default is None.

    Returns:
        np.ndarray: A NumPy array `X_data` of shape (nb_timesteps, len(input_variables), lon_delta_pixels, lat_delta_pixels).
                    Contains the processed data for each variable at each timestep.
    """
    
    # Check if the lists of UTC, latitude, and longitude match the number of timesteps
    if len(utc_list) != nb_timesteps:
        print('ERROR -> len(utc_list)', len(utc_list), 'nb_timesteps', nb_timesteps)
        raise ValueError('len(utc_list) != nb_timesteps')
    if len(lat_list) != nb_timesteps:
        print('ERROR -> len(utc_list)', len(utc_list), 'nb_timesteps', nb_timesteps)
        raise ValueError('len(utc_list) != nb_timesteps')
    
    # List of computed variables that require df_merged_2d_3d
    computed_vars = ['SHEAR', 'SHEARV', 'DEEPSHEAR', 'DEEPSHEARV', 'DIFF_FMSE_MID_BL', 'INT_FMSE_BL', 'INT_FMSE_MID']
    
    # Check if any computed variables are included in the input variables, and require df_merged_2d_3d if so
    if any(var in input_variables for var in computed_vars):
        if df_merged_2d_3d is None:
            raise ValueError(f"'df_merged_2d_3d' is required for one of the variables {computed_vars}, but it is None.")
    
    # Retrieve configuration settings for pixel and degree windows
    lat_delta_pixels = DeepFate.config.SPACE_WINDOW['lat_delta_pixels']
    lon_delta_pixels = DeepFate.config.SPACE_WINDOW['lon_delta_pixels']
    lat_delta_degrees = DeepFate.config.SPACE_WINDOW['lat_delta_degrees']
    lon_delta_degrees = DeepFate.config.SPACE_WINDOW['lon_delta_degrees']
    
    # Initialize X_data array to hold the feature data for all timesteps
    X_data = np.zeros((nb_timesteps, len(input_variables), lon_delta_pixels, lat_delta_pixels))

    # Iterate over each timestep (utc_time, latitude, longitude)
    for i, (utc_time, lat, long) in enumerate(zip(utc_list, lat_list, long_list)):
        
        # Generate paths to data based on the UTC time
        path_dyamond_dir_utc_time = generate_dyamond_file_path_from_utc(utc_time=utc_time, df_relation_table_UTC=df_relation_table)
        path_toocan_file_utc_time = generate_img_seg_file_path_from_utc(utc_time=utc_time, df_relation_table_UTC=df_relation_table)
        
        # Iterate over each variable in the input variables list
        for j, var in enumerate(input_variables):
        
            if var == 'MCS_segmentation':  # Handling segmentation images
                file_path = os.path.join(DeepFate.config.PATH_SEGMENTED_IMGS, path_toocan_file_utc_time)
                X = open_xarray_rolling_lon(file_path=file_path, 
                                            lat_bary=lat, 
                                            lon_bary=long, 
                                            lat_delta=lat_delta_degrees, 
                                            lon_delta=lon_delta_degrees, 
                                            output_shape=(lat_delta_pixels, lon_delta_pixels),
                                            keys_sel=('latitude', 'longitude'))

            elif var in computed_vars:  # Handling computed variables (e.g., shear)
                path_dyamond_dir_computed = df_merged_2d_3d[df_merged_2d_3d['path_dyamond_2d'] == path_dyamond_dir_utc_time]['path_dyamond_3d'].iloc[0]
                file_path = os.path.join(DeepFate.config.PATH_COMPUTED_DYAMOND_2D, f"{path_dyamond_dir_computed}_{var}.nc")
                if os.path.exists(file_path):
                    X = open_xarray_rolling_lon(file_path=file_path, 
                                                lat_bary=lat, 
                                                lon_bary=long, 
                                                lat_delta=lat_delta_degrees, 
                                                lon_delta=lon_delta_degrees, 
                                                output_shape=(lat_delta_pixels, lon_delta_pixels))
                else:
                    X = np.zeros((lat_delta_pixels, lon_delta_pixels))  # Default empty array if file does not exist

            else:  # Handling other standard variables
                file_path = os.path.join(DeepFate.config.PATH_DYAMOND_ROOT_DIR, f"{path_dyamond_dir_utc_time}.{var}.nc")
                X = open_xarray_rolling_lon(file_path=file_path, 
                                            lat_bary=lat, 
                                            lon_bary=long, 
                                            lat_delta=lat_delta_degrees, 
                                            lon_delta=lon_delta_degrees, 
                                            output_shape=(lat_delta_pixels, lon_delta_pixels))
            
            # Check if the resulting data array has the expected shape
            if X.shape != (lat_delta_pixels, lon_delta_pixels):
                raise ValueError(f'X.shape != (lat_delta_pixels, lon_delta_pixels), actual size: {X.shape}')
    
            # Store the data in the X_data array
            X_data[i, j] = X.astype(np.float32)  # Store as float32 for memory efficiency
            
    return X_data


def get_z_data(dict_specs_mcs):
    """
    Constructs a feature vector (z_array) from the provided dictionary of MCS (Mesoscale Convective Systems) specifications.

    The resulting array contains various metrics and statistics that describe the MCS, such as:
    - MCS label
    - Duration of the MCS
    - Average velocity
    - Classification
    - Distance
    - Maximum extension
    - Mean latitude of MCS
    - Mean longitude of MCS

    Args:
        dict_specs_mcs (dict): A dictionary containing specifications of the MCS, including labels, duration, velocity, 
                                classification, distance, extension, and lists of latitude and longitude.

    Returns:
        np.ndarray: A NumPy array containing the specified features in the order:
                    [label, duration, mean_velocity, classification, distance, max_extension, mean_latitude, mean_longitude]
    """
    
    # Create the feature vector (z_array) with the required metrics
    z_array = np.array([
        dict_specs_mcs['label_toocan_mcs'],                      # Label of the MCS
        dict_specs_mcs['duration_mcs'],                          # Duration of the MCS
        np.mean(dict_specs_mcs['average_velocity']).astype(float),  # Mean velocity (converted to float)
        dict_specs_mcs['classif_JIRAK'],                         # Classification (JIRAK)
        dict_specs_mcs['dist'],                                  # Distance (likely from a reference point)
        dict_specs_mcs['max_extend'],                            # Maximum extension of the MCS
        np.mean(dict_specs_mcs['lat_list_mcs']).astype(float),     # Mean latitude (converted to float)
        np.mean(dict_specs_mcs['lon_list_mcs']).astype(float)      # Mean longitude (converted to float)
    ])
    
    return z_array

def get_y_data(object_mcs, start_time=None, end_time=None):
    ### DEPRECIATED -> not used !!
    pass

def precompute_single_mcs(mcs_object, start_time, end_time, df_merged_2d_3d=None):
    """
    Precomputes features for a single MCS object within a specified time range.
    
    Args:
    - mcs_object: MCS object containing data for the convective system
    - start_time: start of the time window
    - end_time: end of the time window
    - df_merged_2d_3d: Optional merged dataframe for 2D/3D data (if needed)
    
    Returns:
    - X: Feature array for input
    - y: Target array for output
    - z: Array containing additional data (e.g., geospatial features)
    """
    df_relation_table = pd.read_csv(DeepFate.config.PATH_RELATION_TABLE)
    dict_specs_mcs = get_specs_mcs(mcs_object, start_time, end_time) 

    assert mcs_object.DCS_number == dict_specs_mcs['label_toocan_mcs']

    try:
        X = get_X_data(
            label_mcs=dict_specs_mcs['label_toocan_mcs'],
            utc_list=dict_specs_mcs['utc_list_mcs'],
            lat_list=dict_specs_mcs['lat_list_mcs'],
            long_list=dict_specs_mcs['lon_list_mcs'],
            df_relation_table=df_relation_table,
            df_merged_2d_3d=df_merged_2d_3d
        )
    except Exception as e:
        print(f"Error getting X data: {e}")
        X = np.zeros((DeepFate.config.NB_TIMESTEPS, len(DeepFate.config.INPUT_VARIABLES), 128, 128))

    y = np.array([
        dict_specs_mcs['duration_mcs'], 
        dict_specs_mcs['max_extend'],
        dict_specs_mcs['delta_time_maximal_extension']
    ])

    z = get_z_data(dict_specs_mcs)

    assert dict_specs_mcs['duration_mcs'] == (end_time - start_time + 1) / 2

    return X, y, z

################### MAIN FUNCTION TO PRECOMPUTE DATASET ###########

def precompute_all_mcs(start_index, stop_index, path_output_h5_file, parallelize: bool = True, nb_workers: int = 64, nb_batchs: int = 100, df_merged_2d_3d=None, mode_3d=None, UTC_3d_start=None, UTC_3d_end=None):
    """
    Precomputes data for multiple MCS objects and saves them to an HDF5 file.

    Parameters:
    -----------
    start_index : int
        The starting index for the subset of MCS objects to process.
    stop_index : int
        The ending index for the subset of MCS objects to process.
    path_output_h5_file : str
        The path where the output HDF5 file will be saved.
    parallelize : bool, optional (default=True)
        Whether to parallelize the computation using multiple workers.
    nb_workers : int, optional (default=64)
        The number of workers to use when parallelizing the computation.
    nb_batchs : int, optional (default=100)
        The number of batches to split the data into when parallelizing.
    df_merged_2d_3d : pd.DataFrame or None, optional (default=None)
        DataFrame containing merged 2D and 3D data. Needed for 3D mode.
    mode_3d : bool or None, optional (default=None)
        Flag indicating if the data should be processed in 3D mode.
    UTC_3d_start : int or None, optional (default=None)
        Start time for the 3D data processing.
    UTC_3d_end : int or None, optional (default=None)
        End time for the 3D data processing.

    Returns:
    --------
    None
        The function writes the precomputed data to an HDF5 file at the specified path.
    """
    
    # Calculate the number of batches based on the range of MCS objects
    nb_batchs = int((stop_index - start_index) / 256) + 1

    # Load and filter valid MCS objects
    path = PATH_TOOCAN_GLOBAL_FILE  # Update with the correct path to the TOOCAN global file
    list_object_mcs = load_TOOCAN_DYAMOND(path)  # Load the TOOCAN DYAMOND data

    list_valid_mcs = get_list_valid_mcs(
        list_object_mcs=list_object_mcs,
        max_area=DeepFate.config.MCS_SPECS_RANGE['max_area'][1],
        min_area=DeepFate.config.MCS_SPECS_RANGE['max_area'][0],
        duration_max=DeepFate.config.MCS_SPECS_RANGE['duration_hour'][1],  # Duration in hours
        duration_min=DeepFate.config.MCS_SPECS_RANGE['duration_hour'][0],
        lat_max=DeepFate.config.MAX_LAT_TROPICS
    )

    # Get the validity and time ranges for the MCS lifecycles
    validity, start_times, end_times = get_validity_lifecycles_start_end(
        list_valid_mcs, mode_3d=mode_3d, UTC_3d_start=UTC_3d_start, UTC_3d_end=UTC_3d_end
    )
    
    # Filter the MCS objects based on validity
    list_valid_mcs_2 = [list_valid_mcs[i] for i in range(len(list_valid_mcs)) if validity[i] is True]
    list_start_times = [start_times[i] for i in range(len(list_valid_mcs)) if validity[i] is True]
    list_end_times = [end_times[i] for i in range(len(list_valid_mcs)) if validity[i] is True]

    # Slice the lists to process only the specified range of MCS objects
    list_valid_mcs_2 = list_valid_mcs_2[start_index:stop_index]
    list_start_times = list_start_times[start_index:stop_index]
    list_end_times = list_end_times[start_index:stop_index]

    # Configuration for time steps and input variables
    nb_timesteps = DeepFate.config.NB_TIMESTEPS
    input_variables = INPUT_VARIABLES
    space_window_pixels = (DeepFate.config.SPACE_WINDOW['lon_delta_pixels'], DeepFate.config.SPACE_WINDOW['lat_delta_pixels'])

    # Define the shapes for the output datasets
    X_full_shape = (len(list_valid_mcs_2), nb_timesteps, len(input_variables), space_window_pixels[0], space_window_pixels[1])
    y_full_shape = (len(list_valid_mcs_2), 3)
    z_full_shape = (len(list_valid_mcs_2), 8)

    # Open the HDF5 file for writing the precomputed data
    with h5py.File(path_output_h5_file, 'w') as hf:
        X_dataset = hf.create_dataset('X', X_full_shape, chunks=((1,) + tuple(X_full_shape[1:])))
        y_dataset = hf.create_dataset('y', y_full_shape)
        z_dataset = hf.create_dataset('z', z_full_shape)

        # If not parallelizing, process each MCS object sequentially
        if not parallelize:
            for i, (object_mcs, start_time, end_time) in tqdm.tqdm(enumerate(zip(list_valid_mcs_2, list_start_times, list_end_times)), total=len(list_valid_mcs_2)):
                # Precompute the data for a single MCS object
                X_i, y_i, z_i = precompute_single_mcs(mcs_object=object_mcs, start_time=start_time, end_time=end_time, df_merged_2d_3d=df_merged_2d_3d)
                X_dataset[i] = X_i
                y_dataset[i] = y_i
                z_dataset[i] = z_i

        else:
            # Split the MCS objects into batches for parallel processing
            splitted_mcs_list = np.array_split(list_valid_mcs_2, nb_batchs)
            splitted_start_list = np.array_split(list_start_times, nb_batchs)
            splitted_end_list = np.array_split(list_end_times, nb_batchs)
            splitted_idx = np.array_split(np.arange(len(list_valid_mcs_2)), nb_batchs)

            # Process each batch in parallel
            for j, (batch_mcs, batch_start_times, batch_end_times, batch_idx) in enumerate(zip(splitted_mcs_list, splitted_start_list, splitted_end_list, splitted_idx)):
                input_precomputing = [(a, b, c, df_merged_2d_3d) for a, b, c in zip(batch_mcs, batch_start_times, batch_end_times)]

                print(f'Processing batch {j + 1} of {nb_batchs}')

                # Use a Pool of workers to parallelize the computation for each batch
                with Pool(nb_workers) as p:
                    outputs = p.starmap(precompute_single_mcs, tqdm.tqdm(input_precomputing))

                    # Write the results for the current batch to the HDF5 file
                    X_dataset[batch_idx] = np.array([out[0] for out in outputs]).astype(np.float32)
                    y_dataset[batch_idx] = np.array([out[1] for out in outputs]).astype(np.float16)
                    z_dataset[batch_idx] = np.array([out[2] for out in outputs]).astype(np.float32)

            
