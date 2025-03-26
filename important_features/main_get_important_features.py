import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm


import sys
sys.path.append('/home/b/b381993/')
from DeepFate.model.utils_model import get_datasets
from DeepFate.important_features.utils_important_features import get_features_importance, get_variances

# Define paths
save_path = '/work/bb1153/b381993/data/VERSION_FEBRUARY_2025_3D_for_september_systems/experiments_models'
save_path_important_features = '/work/bb1153/b381993/data/VERSION_FEBRUARY_2025_3D_for_september_systems/important_features'
root_dir = '/work/bb1153/b381993/data/VERSION_FEBRUARY_2025_3D_for_september_systems/'

# Load train and test datasets
df_train = pd.read_csv(os.path.join(root_dir, 'fusion_train_dataset.csv'))
df_test = pd.read_csv(os.path.join(root_dir, 'fusion_test_dataset.csv'))

#experiment 1: Only growth rate 
dict_exp1 = {'df_train':df_train, 'df_test':df_test, 'all_features':False, 'one_features_name':'gradient_area'}

#experiment 2: everything except growth rate, shear and fmse
blacklist_exp2=['gradient_area',  'average_diameter','mcs_area', 'var_13', 'var_14', 'var_15', 'var_16', 'var_17', 'var_18', 'var_19']
dict_exp2 = {'df_train':df_train, 'df_test':df_test, 'all_features':True, 'one_features_name':'', 'blacklist':blacklist_exp2}

#experiment 3: everything except growth rate and all related to size
blacklist_exp3=['gradient_area', 'average_diameter', 'mcs_area']
dict_exp3 = {'df_train':df_train, 'df_test':df_test, 'all_features':True, 'one_features_name':'', 'blacklist':blacklist_exp3}

#experiment 4: everything but growth rate
blacklist_exp4=['gradient_area', 'mcs_area']
dict_exp4 = {'df_train':df_train, 'df_test':df_test, 'all_features':True, 'one_features_name':'', 'blacklist':blacklist_exp4}

#experiment 5: everything
blacklist_exp5=[]
dict_exp5 = {'df_train':df_train, 'df_test':df_test, 'all_features':True, 'one_features_name':'', 'blacklist':blacklist_exp5}


#experiment 5: everything
blacklist_exp6=['gradient_area', 'average_diameter', 'var_1_', 'var_2', 'var_3', 'var_4', 'var_7', 'var_8', 'var_10', 'var_11', 'var_17', 'var_18']
dict_exp6 = {'df_train':df_train, 'df_test':df_test, 'all_features':True, 'one_features_name':'', 'blacklist':blacklist_exp6}

i_values = [5, 6, 9, 12, 13, 14, 15, 16, 19]
std_everywhere = [f"std_everywhere_var_{i}" for i in i_values]
mean_under_cloud = [f"mean_under_cloud_var_{i}" for i in i_values]
wantlist_exp7=std_everywhere + mean_under_cloud
wantlist_exp7.extend(['average_diameter', 'interaction_power', 'eccentricity'])
dict_exp7 = {'df_train':df_train, 'df_test':df_test, 'all_features':True, 'one_features_name':'', 'wantlist':wantlist_exp7}



# Experiment settings
exp = 'EXP4'
dict_exp = dict_exp4
nt_eval = 3

for exp, dict_exp in tqdm(zip(['EXP2', 'EXP3', 'EXP4', 'EXP5', 'EXP6', 'EXP7'], [dict_exp2, dict_exp3, dict_exp4, dict_exp5, dict_exp6, dict_exp7])):
#for exp, dict_exp in tqdm(zip(['EXP7'], [dict_exp7])):

    # Get the datasets for the experiment
    if exp!='EXP7':
        train_dataset, _ = get_datasets(
            all_features=dict_exp['all_features'], 
            df_train=dict_exp['df_train'], 
            df_test=dict_exp['df_test'], 
            one_features_name=dict_exp['one_features_name'], 
            blacklist=dict_exp['blacklist'], 
            nb_timesteps=nt_eval
        )
    else:
        train_dataset, _ = get_datasets(
            all_features=dict_exp['all_features'], 
            df_train=dict_exp['df_train'], 
            df_test=dict_exp['df_test'], 
            one_features_name=dict_exp['one_features_name'], 
            wantlist=dict_exp['wantlist'], 
            nb_timesteps=nt_eval
        )

    # Get important features using Lasso model
    features_selected_names, values_features, index_argsort = get_features_importance(
        model_str='Lasso', 
        path_model=save_path, 
        df_train=train_dataset, 
        nb_timesteps=nt_eval, 
        exp=exp
    )

    # Load the mapping of feature names for plotting
    df_names = pd.read_csv('/home/b/b381993/DeepFate/notebooks/name_dict_new_var.csv')

    # Safely retrieve new feature names for plotting, handling missing values
    new_features_selected_names = [
        df_names[df_names['Old Name'] == feature]['Name Plot'].values[0] 
        if not df_names[df_names['Old Name'] == feature].empty else feature 
        for feature in features_selected_names
    ]

    # Save important feature names and values
    np.savetxt(os.path.join(save_path_important_features, f'new_features_selected_names_nt_{nt_eval}_{exp}.txt'), 
            new_features_selected_names[:15], fmt='%s')
    np.savetxt(os.path.join(save_path_important_features, f'features_selected_names_nt_{nt_eval}_{exp}.txt'), 
            features_selected_names[:15], fmt='%s')
    np.savetxt(os.path.join(save_path_important_features, f'values_features_nt_{nt_eval}_{exp}.txt'), 
            values_features[:15])

    # Preprocess train data (only select the top 15 features)
    df_train_cut = df_train[features_selected_names[:15]]
    scaler = StandardScaler().fit(df_train_cut.values)
    X_train = scaler.transform(df_train_cut.values)
    y_train = df_train['y_max_extend']

    # Save scaled train data and target values
    np.savetxt(os.path.join(save_path_important_features, 'y_train.txt'), y_train)

    # Separate features into system and environment based on the '*' in names
    idx_sys = np.array(['*' in item for item in new_features_selected_names[:15]])
    sys = np.where(idx_sys)[0]
    env = np.where(~idx_sys)[0]

    coef_i = values_features[:15]

    # Calculate system and environment components using matrix operations
    syst_component = X_train[:, sys].dot(coef_i[sys])
    env_component = X_train[:, env].dot(coef_i[env])

    # Save system and environment components
    np.savetxt(os.path.join(save_path_important_features, f'syst_component_nt{nt_eval}_{exp}.txt'), syst_component)
    np.savetxt(os.path.join(save_path_important_features, f'env_component_nt{nt_eval}_{exp}.txt'), env_component)

    ######### Variance Calculation
    # Define small and big categories based on threshold of y_max_extend
    idx_small = np.where(y_train.values < 120)[0]
    idx_big = np.where(y_train.values >= 120)[0]

    X_train_small = X_train[idx_small, :]
    X_train_big = X_train[idx_big, :]

    # Calculate variances for all, small, and big datasets
    dict_variances_all = get_variances(X_train, coef_i, sys, env)
    dict_variances_small = get_variances(X_train_small, coef_i, sys, env)
    dict_variances_big = get_variances(X_train_big, coef_i, sys, env)

    # Save variances in pickle format
    with open(os.path.join(save_path_important_features, f'dict_variances_all_{nt_eval}_{exp}.pkl'), 'wb') as f:
        pickle.dump(dict_variances_all, f)
    with open(os.path.join(save_path_important_features, f'dict_variances_small_{nt_eval}_{exp}.pkl'), 'wb') as f:
        pickle.dump(dict_variances_small, f)
    with open(os.path.join(save_path_important_features, f'dict_variances_big_{nt_eval}_{exp}.pkl'), 'wb') as f:
        pickle.dump(dict_variances_big, f)

