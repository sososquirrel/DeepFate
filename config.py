import numpy as np


## PATH RAW DATA
PATH_TOOCAN_GLOBAL_FILE = '/work/bb1153/b381993/data3/data/new_TOOCAN-GLOBAL-20160901-20160910.dat.gz'
#PATH_TOOCAN_GLOBAL_FILE ='/work/bb1153/b381993/data3/data/new_TOOCAN-GLOBAL-20160801-20160831.dat.gz'
PATH_SEGMENTED_IMGS = '/work/bb1153/b381993/data3/data/'
PATH_RELATION_TABLE ='/work/bb1153/b381993/data3/data/RRR_utlimate_relation_2_table_UTC_dyamond_segmentation.csv'


PATH_DYAMOND_ROOT_DIR='/fastdata/ka1081/DYAMOND/data/summer_data/SAM-4km/OUT_2D/'



## CONSTANTS
MIN_LONGITUDE_DYAMOND = 1.953125e-02
MAX_LONGITUDE_DYAMOND = 3.599805e+02
MAX_LAT_TROPICS = 30
FORBIDDEN_UTC = 17054.01

## VARS MCS
NB_TIMESTEPS = 10 # previously 10

INPUT_VARIABLES = ['MCS_segmentation', 'LWNTA.2D','PW.2D','RH500.2D','RH700.2D','T2mm.2D',
                   'IWP.2D','U10m.2D','V10m.2D', 'LANDMASK.2D', 'OM500.2D', 'OM700.2D', 'OM850.2D']#, 'Precac']

OUTPUT_VARIABLES = np.array(['duration', 'time_maximal_extension', 'maximal_extension'])

## FILTERING OPTION FOR MCS
MCS_SPECS_RANGE = {'max_area' : (1000,400000),
                   'duration_hour' : (4.5, 100.0), 
                  't_max_area' : (0.0, 70.0)}
SPACE_WINDOW = {'lat_delta_degrees' : 2.5, 'lon_delta_degrees' : 2.5,
                'lat_delta_pixels':128, 'lon_delta_pixels':128}

### for validity filtering
ROLLING_WINDOW=4
GRADIENT_THRESHOLD=600
FRACTION_MAX_END=0.4
FRACTION_MAX_START=0.5


## GAUSSIAN FILTER FOR MCS MASK
t_arr = np.linspace(-10, 10, 10)
BUMP = np.exp(-0.02*t_arr**2)
BUMP /= np.trapz(BUMP) # normalize the integral to 1
