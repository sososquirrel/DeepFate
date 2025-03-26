

RELATION_TABLE_2D = '/work/bb1153/b381993/data3/data/RRR_utlimate_relation_2_table_UTC_dyamond_segmentation.csv'
RELATION_TABLE_3D = '/work/bb1153/b381993/data3/data/relation_table_UTC_2d_3d_retrieve_filesv_2.csv'
MERGED_TABLE_OLD = '/work/bb1153/b381993/data3/data/MERGED_RALATION_TABLE_2D_3D_IMGSEG.csv'
MERGED_TABLE_OLD_OLD = '/work/bb1153/b381993/data3/data/MERGED_RALATION_TABLE_2D_3D.csv'

#MERGED_TABLE = '/work/bb1153/b381993/data3/data/MERGED_RELATION_3D_2D_ALL_FILES.csv'

MERGED_TABLE = '/work/bb1153/b381993/data3/data/MERGED_RELATION_3D_2D_ALL_FILES_no_duplicate.csv'


FOLDER_PATH = '/work/bb1153/b381993/computed_dyamond_2d'


ROOT_DICT_PATHS = {'root_toocan': '/work/bb1153/b381993/data3/data/',
'root_2d_files': '/fastdata/ka1081/DYAMOND/data/summer_data/SAM-4km/OUT_2D/',
'root_3d_files': '/fastdata/ka1081/DYAMOND/data/summer_data/SAM-4km/OUT_3D/'}

# Constants
SPECIFIC_HEAT_CAPACITY = 1005  # J/kg/K for dry air at constant pressure (cp)
LATENT_HEAT_VAPORIZATION = 2.5e6  # J/kg for water (Lv)
GRAVITY = 9.18  # m/s^2 (gravity constant)
WINDOW_SIZE_X = 10  # Filter window size in x dimension
WINDOW_SIZE_Y = 10  # Filter window size in y dimension
BOUNDARY_LAYER_HEIGHT = 1000  # Boundary layer top height in meters (zbl)
LOW_MID_TROPOSPHERE = 4000  #  low level of mid troposphere in meters (zbl)
HIGH_MID_TROPOSPHERE = 6000  #  low level of mid troposphere in meters (zbl)

FMSE_CAP = [-6000, 6000]  # Cap for FMSE integrals