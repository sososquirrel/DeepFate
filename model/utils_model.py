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



def get_X_cols(columns, nb_timesteps):
    output_cols_1 = []
    for col in columns:
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
                 nb_timesteps: int, one_features_name: str):

    
    if all_features:
        cols = get_X_cols(df_train.columns, nb_timesteps=nb_timesteps)
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
        'MLPRegressor': MLPRegressor(),
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
               folder_path:str = '/work/bb1153/b381993/data'):
    
            ##get train_dataset and scale it 
        model = get_model(model=model_str)
        train_dataset, _ = get_datasets(all_features, df_train, df_test, time_selected, one_features_name)
        scaler = StandardScaler().fit(train_dataset.values)
        X_train = scaler.transform(train_dataset.values)
        y_train = df_train['y_max_extend']
        
        ##fit the train dataset
        model.fit(X_train, y_train)
        
        ##save the model
        path_to_save = os.path.join(folder_path, f'{model_str}_{output_path_model}_{time_selected}.joblib')
        joblib.dump(model,path_to_save)
        return None
    

def train_model(df_train: pd.DataFrame,
                df_test: pd.DataFrame,
                all_features: bool = False,
                model_str:str='Lasso',     
                output_path_model: str = f'ONLY_GROWTH_RATE',
                list_timesteps: list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                one_features_name: str = 'gradient_area',
               folder_path:str = '/work/bb1153/b381993/data'):
    
    num_processes = min(multiprocessing.cpu_count(), len(list_timesteps))
    
    additional_kwargs = {'df_train' : df_train,
                'df_test' : df_test,
                'all_features' : all_features,
                'model_str'  : model_str,  
                'output_path_model' : output_path_model,
                'one_features_name' : one_features_name,
               'folder_path' : folder_path}

    with multiprocessing.Pool(processes=num_processes) as pool:
        results = list(pool.imap(partial(train_single_timestep_model, **additional_kwargs), list_timesteps))
    

def predict_model(df_train: pd.DataFrame,
                df_test: pd.DataFrame,
                  output_path_model: str = f'ONLY_GROWTH_RATE',
                          model_str:str='Lasso',
                        all_features: bool = False,
                         list_timesteps: list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                         one_features_name: str = 'gradient_area',
                  folder_path:str = '/work/bb1153/b381993/data'):
    
    y_test_list, y_preds_list = [], []
    
    for nb_timesteps in tqdm(list_timesteps, total=len(list_timesteps)):
        ##get model
        path_to_load = os.path.join(folder_path, f'{model_str}_{output_path_model}_{nb_timesteps}.joblib')
        model = joblib.load(path_to_load)
        
        ##get test_dataset, and scale it with with the train
        train_dataset, test_dataset = get_datasets(all_features, df_train, df_test, nb_timesteps, one_features_name)
        scaler = StandardScaler().fit(train_dataset.values)
        X_test = scaler.transform(test_dataset.values)
        y_test = df_test['y_max_extend'].values
        
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
                         folder_path:str = '/work/bb1153/b381993/data'):
    pearsonr_list, rmse_list = [], []
    for nb_timesteps in tqdm(list_timesteps, total=len(list_timesteps)):
        
        ##get the model
        model = get_model(model=model_str)
        path_to_load = os.path.join(folder_path, f'{model_str}_{output_path_model}_{nb_timesteps}.joblib')
        model = joblib.load(path_to_load)
        
        ##get test_dataset, and scale it with with the train
        train_dataset, test_dataset = get_datasets(all_features, df_train, df_test, nb_timesteps, one_features_name)
        scaler = StandardScaler().fit(train_dataset.values)
        X_test = scaler.transform(test_dataset.values)
        y_test = df_test['y_max_extend']
        
        ##get the prediction
        y_preds = model.predict(X_test)
        
        ##get metrics 
        pearson_r = pearsonr(y_preds, y_test)
        rmse = np.sqrt(mean_squared_error(y_preds, y_test))
        
        ##store metrics
        pearsonr_list.append(pearson_r[0])
        rmse_list.append(rmse)
        
    return pearsonr_list, rmse_list


