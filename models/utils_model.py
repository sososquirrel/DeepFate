import pandas as pd
import os
from functools import partial
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectKBest, chi2, SelectFromModel
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPRegressor
from sklearn.feature_selection import SequentialFeatureSelector
from scipy.stats import pearsonr
from tqdm import tqdm
import joblib
from scipy.stats import gaussian_kde
from scipy.stats import percentileofscore

import pandas as pd
from sklearn.linear_model import Lasso, HuberRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor

import numpy as np

import DeepFate
from DeepFate.config import *

import multiprocessing



def get_X_cols(columns, nb_timesteps, wantlist=None, blacklist=[]):
    output_cols_1 = []
    
    # Convert blacklist to set for faster lookup
    blacklist = set(blacklist)

    # If wantlist is provided, convert to a set; if not, ignore the wantlist
    if wantlist:
        wantlist = set(wantlist)
        
        # Check for conflicts between wantlist and blacklist
        conflicting_cols = wantlist & blacklist
        if conflicting_cols:
            print(f"Warning: The following columns are in both the wantlist and blacklist and will be excluded: {conflicting_cols}")
            # Remove conflicting columns from both lists
            wantlist -= conflicting_cols
            blacklist -= conflicting_cols
    else:
        wantlist = None

    for col in columns:
        # Skip columns that start with any item in the blacklist
        if any([bl in col for bl in blacklist]):
            continue
        
        # If wantlist is provided, check if the column is in the wantlist
        if wantlist is not None:
            if any([wl in col for wl in wantlist]) and any([col.endswith(f'time_{i}') for i in range(nb_timesteps)]):
                output_cols_1.append(col)
        else:
            # If no wantlist is provided, only check the timestep condition
            if any([col.endswith(f'time_{i}') for i in range(nb_timesteps)]):
                output_cols_1.append(col)
    
    return output_cols_1


def get_one_feature(df, feature_name):
    output_col = []
    for col in df.columns:
        if col.startswith(f'{feature_name}'):
            output_col.append(col)

    
    return output_col

def get_datasets(all_features: bool, df_train: pd.DataFrame, df_test: pd.DataFrame,
                 nb_timesteps: int, one_features_name: str, blacklist=[], wantlist=None):

    
    if all_features:
        cols = get_X_cols(df_train.columns, nb_timesteps=nb_timesteps, blacklist=blacklist, wantlist=wantlist)
    else:
        cols = get_one_feature(df_train, one_features_name)
        cols = get_X_cols(cols, nb_timesteps=nb_timesteps)
    
    train_dataset = df_train[cols]
    test_dataset = df_test[cols]

    return train_dataset, test_dataset


def get_model(model: str = 'Lasso'):
    models = {
        'Lasso': Lasso(),
        'RandomForest': RandomForestRegressor(),
        'MLPRegressor':MLPRegressor(hidden_layer_sizes=(100,), max_iter=1000, alpha=0.001, random_state=42, early_stopping=True, verbose=False)
,
        'HuberRegressor':HuberRegressor()
    }
    if model in models:
        return models[model]
    else:
        raise ValueError(f"Invalid model name. Supported models are {', '.join(models.keys())}.")

def train_single_timestep_model(time_selected:int,
                                df_train: pd.DataFrame,
                                df_test: pd.DataFrame,
                                all_features: bool = False,
                                model_str:str='Lasso',     
                                output_path_model: str = f'ONLY_GROWTH_RATE',
                                one_features_name: str = 'gradient_area',
                            folder_path:str = '/work/bb1153/b381993/data',
                            blacklist=[],
                            wantlist=None,
                            no_corr_input=False,
                            corr_threshold = 0.9,
                            priority_vars=None):
        
        #print(f't{time_selected}')

        ##get train_dataset and scale it 
        model = get_model(model=model_str)

        #print('**model')

        train_dataset, _ = get_datasets(all_features, df_train, df_test, time_selected, one_features_name, blacklist=blacklist, wantlist=wantlist)
        
        if no_corr_input:
            to_drop = remove_correlated_features(train_dataset, 
                               corr_threshold= corr_threshold, 
                               priority_vars= priority_vars)
            
            train_dataset=train_dataset.drop(columns=to_drop)

        
        scaler = StandardScaler().fit(train_dataset.values)

        X_train = scaler.transform(train_dataset.values)
        y_train = df_train['y_max_extend']
        #y_train=df_train['Sqrt_surfmaxkm2_235K']
        
        ##fit the train dataset
        model.fit(X_train, y_train)
        
        ##save the model
        if not no_corr_input:
            path_to_save = os.path.join(folder_path, f'{model_str}_{output_path_model}_{time_selected}.joblib')
        else:
            path_to_save = os.path.join(folder_path, f'{model_str}_{output_path_model}_{time_selected}_presel.joblib')

        #print('!!',path_to_save )
        joblib.dump(model,path_to_save)
        #print('****', path_to_save)
        return None
    

def train_model(df_train: pd.DataFrame,
                df_test: pd.DataFrame,
                all_features: bool = False,
                model_str:str='Lasso',     
                output_path_model: str = 'ONLY_GROWTH_RATE',
                list_timesteps: list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                one_features_name: str = 'gradient_area',
               folder_path:str = '/work/bb1153/b381993/data',
               blacklist=[],
               wantlist=None,
               no_corr_input=False,
               corr_threshold = 0.9,
               priority_vars=None):
    
    num_processes = min(multiprocessing.cpu_count(), len(list_timesteps))
    print(num_processes)
    additional_kwargs = {'df_train' : df_train,
                'df_test' : df_test,
                'all_features' : all_features,
                'model_str'  : model_str,  
                'output_path_model' : output_path_model,
                'one_features_name' : one_features_name,
               'folder_path' : folder_path,
               'blacklist' : blacklist,
               'wantlist':wantlist,
               'no_corr_input':no_corr_input,
               'corr_threshold':corr_threshold,
               'priority_vars':priority_vars}
    
    with multiprocessing.Pool(processes=num_processes) as pool:
        results = list(tqdm(pool.imap(partial(train_single_timestep_model, **additional_kwargs), list_timesteps)))
    
 

def predict_model(df_train: pd.DataFrame,
                df_test: pd.DataFrame,
                  output_path_model: str = f'ONLY_GROWTH_RATE',
                          model_str:str='Lasso',
                        all_features: bool = False,
                         list_timesteps: list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                         one_features_name: str = 'gradient_area',
                  folder_path:str = '/work/bb1153/b381993/data',
                  blacklist=[],
                  wantlist=None,
                  no_corr_input=False,
                corr_threshold = 0.9,
                priority_vars=None):
    
    y_test_list, y_preds_list = [], []
    
    for nb_timesteps in tqdm(list_timesteps, total=len(list_timesteps)):
        ##get model
        if not no_corr_input:
            path_to_load = os.path.join(folder_path, f'{model_str}_{output_path_model}_{nb_timesteps}.joblib')
        else:
            path_to_load = os.path.join(folder_path, f'{model_str}_{output_path_model}_{nb_timesteps}_presel.joblib')

        model = joblib.load(path_to_load)
        
        ##get test_dataset, and scale it with with the train
        train_dataset, test_dataset = get_datasets(all_features, df_train, df_test, nb_timesteps, one_features_name, blacklist=blacklist, wantlist=wantlist)
        
        if no_corr_input:
            to_drop = remove_correlated_features(train_dataset, 
                               corr_threshold= corr_threshold, 
                               priority_vars= priority_vars)
            
            train_dataset=train_dataset.drop(columns=to_drop)
            test_dataset = test_dataset.drop(columns=to_drop) 
        
        
        scaler = StandardScaler().fit(train_dataset.values)
        X_test = scaler.transform(test_dataset.values)
        y_test = df_test['y_max_extend'].values
        #y_test = df_test['Sqrt_surfmaxkm2_235K'].values
        
        ##get the prediction
        y_preds = model.predict(X_test)
        
        ##store result
        y_test_list.append(y_test)
        y_preds_list.append(y_preds)
        
    return y_test_list, y_preds_list
    
    
        
def evaluate_lasso_model(df_train: pd.DataFrame,
                         df_test: pd.DataFrame,
                         output_path_model: str = f'ONLY_GROWTH_RATE',
                        all_features: bool = False,
                         model_str:str='Lasso',
                         list_timesteps: list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                         one_features_name: str = 'gradient_area',
                         folder_path:str = '/work/bb1153/b381993/data',
                         blacklist=[],
                         wantlist=None,
                        no_corr_input=False,
                        corr_threshold = 0.9,
                        priority_vars=None):
    
    pearsonr_list, rmse_list = [], []
    for nb_timesteps in tqdm(list_timesteps, total=len(list_timesteps)):
        
        ##get the model
        ##get model
        if not no_corr_input:
            path_to_load = os.path.join(folder_path, f'{model_str}_{output_path_model}_{nb_timesteps}.joblib')
        else:
            path_to_load = os.path.join(folder_path, f'{model_str}_{output_path_model}_{nb_timesteps}_presel.joblib')

        model = joblib.load(path_to_load)
        
        ##get test_dataset, and scale it with with the train
        train_dataset, test_dataset = get_datasets(all_features, df_train, df_test, nb_timesteps, one_features_name, blacklist=blacklist, wantlist=wantlist)
        
        if no_corr_input:
            to_drop = remove_correlated_features(train_dataset, 
                               corr_threshold= corr_threshold, 
                               priority_vars= priority_vars)
            
            train_dataset=train_dataset.drop(columns=to_drop)
            test_dataset = test_dataset.drop(columns=to_drop) 
        
        scaler = StandardScaler().fit(train_dataset.values)
        X_test = scaler.transform(test_dataset.values)
        y_test = df_test['y_max_extend']
        #y_test = df_test['Sqrt_surfmaxkm2_235K']
        
        
        ##get the prediction
        y_preds = model.predict(X_test)
        
        ##get metrics 
        pearson_r = pearsonr(y_preds, y_test)
        rmse = np.sqrt(mean_squared_error(y_preds, y_test))
        
        ##store metrics
        pearsonr_list.append(pearson_r[0])
        rmse_list.append(rmse)
        
    return pearsonr_list, rmse_list



def get_features_importance(df_train: pd.DataFrame,
                         df_test: pd.DataFrame,
                         output_path_model: str = f'ONLY_GROWTH_RATE',
                        all_features: bool = False,
                         model_str:str='Lasso',
                         nb_timesteps: int = 4,
                         one_features_name: str = 'gradient_area',
                         folder_path:str = '/work/bb1153/b381993/data',
                         blacklist=[]):
    
    
    ##get the model
    model = get_model(model=model_str)
    path_to_load = os.path.join(folder_path, f'{model_str}_{output_path_model}_{nb_timesteps}.joblib')
    model = joblib.load(path_to_load)
    
    ##get test_dataset, and scale it with with the train
    #train_dataset, _ = get_datasets(all_features, df_train, df_test, nb_timesteps, one_features_name, blacklist=blacklist)
    
    if model_str=='Lasso':
        feature_importances = model.coef_
        feature_importances_abs = np.abs(model.coef_)
    elif model_str=='RandomForest':
        feature_importances = model.feature_importances_
        feature_importances_abs = np.abs(model.feature_importances_)
        
    index_argsort = np.argsort(feature_importances_abs)[::-1]
    values_features = feature_importances[index_argsort]
    

    features_selected_names = df_train.columns[index_argsort]
    
    return features_selected_names, values_features, index_argsort


def remove_correlated_features(train_dataset: pd.DataFrame, 
                               corr_threshold: float, 
                               priority_vars: list = None) -> pd.DataFrame:
    """
    Removes features with high cross-correlation, prioritizing features from `priority_vars`.
    
    Parameters:
    - train_dataset: pd.DataFrame, the dataset to filter.
    - corr_threshold: float, the threshold above which features are considered highly correlated.
    - priority_vars: list, the list of variables to prioritize and keep in case of correlations.
    
    Returns:
    - train_dataset: pd.DataFrame, the filtered dataset with less correlated features.
    """
    
    # Compute the correlation matrix
    corr_matrix = train_dataset.corr().abs()
    
    # Select upper triangle of the correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
    # Prepare to track which columns to drop, but give priority to certain variables
    to_drop = []
    priority_correlated_groups = []

    # Loop through the columns and identify the ones to drop based on correlation
    for column in upper.columns:
        # Find highly correlated variables
        correlated_features = upper[column][upper[column] > corr_threshold].index.tolist()

        # If the current column is highly correlated with others and is in the priority list
        if correlated_features and priority_vars and column in priority_vars:
            correlated_priority_vars = [col for col in correlated_features if col in priority_vars]
            
            if len(correlated_priority_vars) > 1:
                # Handle correlated variables within the priority list
                print(f"Correlated priority variables: {correlated_priority_vars}")
                priority_correlated_groups.append(correlated_priority_vars)
            else:
                print(f"Keeping priority variable: {column}")
        elif any(upper[column] > corr_threshold):
            # If not in priority list and highly correlated, mark it for dropping
            to_drop.append(column)

    # If there are correlated groups within priority variables, select representative ones
    for group in priority_correlated_groups:
        # Here, select the variable with the highest variance from the correlated group
        representative_var = max(group, key=lambda x: train_dataset[x].var())
        group_to_drop = [var for var in group if var != representative_var]
        print(f"Retaining {representative_var} from group {group}, dropping {group_to_drop}")
        to_drop.extend(group_to_drop)

    print(f"Dropping {len(to_drop)} highly correlated features: {to_drop}")
    
    # Drop the highly correlated features
    #train_dataset = train_dataset.drop(columns=to_drop)
    
    return to_drop
