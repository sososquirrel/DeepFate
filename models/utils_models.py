import pandas as pd
import sys
module_dir = '/home/b/b381993'
sys.path.append(module_dir)
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


def remove_highly_correlated_features(train_dataset: pd.DataFrame, 
                                      corr_threshold: float, 
                                      priority_vars: list = [],
                                      blacklist_vars:list=[]) -> set:
    """
    Removes features with high cross-correlation, prioritizing features from `priority_vars`.
    
    Parameters:
    - train_dataset: pd.DataFrame, the dataset to filter.
    - corr_threshold: float, the threshold above which features are considered highly correlated.
    - priority_vars: list, the list of variables to prioritize and keep in case of correlations.
    
    Returns:
    - to_drop: set, the columns that should be dropped due to high correlation.
    """
    
    # Ensure blacklist variables exist in the dataset
    blacklist_vars = [col for col in train_dataset.columns if any(var in col for var in blacklist_vars)]


    print(f"Number of columns before blacklist removal: {len(train_dataset.columns)}")
    train_dataset = train_dataset.drop(columns=blacklist_vars)
    print(f"Number of columns after blacklist removal: {len(train_dataset.columns)}")

    # Compute the correlation matrix
    corr_matrix = train_dataset.corr().abs()

    #sns.heatmap(corr_matrix[:100,:100])
    
    # Select upper triangle of the correlation matrix
    #upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
    #upper = upper.dropna()
    # Set to store columns to drop
    to_drop = set()
    for var in blacklist_vars:
        to_drop.add(var)

    # Iterate through the correlation matrix to find highly correlated features
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            # Check if the absolute correlation exceeds the threshold
            if corr_matrix.iloc[i, j] > corr_threshold:
                col_i = corr_matrix.columns[i]
                col_j = corr_matrix.columns[j]

                # Handle the case with priority vars
                # Handle the case with priority vars as substrings
                if any(prio in col_i for prio in priority_vars) and any(prio in col_j for prio in priority_vars):
                    # Both are in priority list (by substring match), keep both or handle separately (custom rule)
                    #print(f"Both {col_i} and {col_j} are in priority vars (substring match), keeping both")
                    None
                elif any(prio in col_i for prio in priority_vars):
                    # Keep the priority variable col_i, drop col_j
                    to_drop.add(col_j)
                    #print(f"Keeping priority variable: {col_i}, dropping correlated: {col_j}")
                elif any(prio in col_j for prio in priority_vars):
                    # Keep the priority variable col_j, drop col_i
                    to_drop.add(col_i)
                    #print(f"Keeping priority variable: {col_j}, dropping correlated: {col_i}")
                else:
                    # Neither variable is in priority list, follow original logic and drop col_j
                    to_drop.add(col_j)
                    #print(f"Neither {col_i} nor {col_j} are in priority vars, dropping: {col_j}")

    return to_drop
