import argparse
from sklearn.feature_selection import GenericUnivariateSelect, f_regression
from sklearn.linear_model import Lasso
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import sys
module_dir = '/home/b/b381993'
sys.path.append(module_dir)
from DeepFate.models.RemoveCorrelatedFeatures import RemoveCorrelatedFeatures
import os
from DeepFate.important_features.utils_important_features import get_datasets
import pandas as pd
import joblib
from scipy.stats import pearsonr
from DeepFate.important_features.utils_important_features import get_datasets
from tqdm import tqdm
from sklearn.metrics import mean_squared_error
import numpy as np

# Setup argument parser
def parse_args():
    parser = argparse.ArgumentParser(description="Train and evaluate Lasso model.")
    parser.add_argument('--train_data', type=str, default='/work/bb1153/b381993/data/VERSION_MARCH_NEW_INTERPRETABLE/train_dataset.csv', 
                        help="Path to the training dataset")
    parser.add_argument('--test_data', type=str, default='/work/bb1153/b381993/data/VERSION_MARCH_NEW_INTERPRETABLE/test_dataset.csv', 
                        help="Path to the test dataset")
    parser.add_argument('--model_save_path', type=str, default='/home/b/b381993/DeepFate/models/saved_models/', 
                        help="Path to save the model")
    parser.add_argument('--metrics_save_path', type=str, default='/home/b/b381993/DeepFate/models/saved_metrics/', 
                        help="Path to save the metrics")
    return parser.parse_args()

# Load datasets
def load_datasets(train_data_path, test_data_path):
    train_dataset_raw = pd.read_csv(train_data_path)
    test_dataset_raw = pd.read_csv(test_data_path)
    return train_dataset_raw, test_dataset_raw

# Save metrics to file
def save_metrics(metrics, save_path, filename):
    np.savetxt(os.path.join(save_path, filename), metrics, delimiter=',')

if __name__ == '__main__':
    # Parse command-line arguments
    args = parse_args()
    
    # Load datasets
    train_dataset_raw, test_dataset_raw = load_datasets(args.train_data, args.test_data)

    print(len(train_dataset_raw), len(test_dataset_raw))

    only_growth_rate = True

    if only_growth_rate:
        dict_exp = {'df_train': train_dataset_raw, 
                    'df_test': test_dataset_raw, 
                    'all_features': False, 
                    'one_features_name': 'gradient_area',
                    'blacklist': []}
    else:
        blacklist_exp = ['gradient_area', 'average_diameter', 'mcs_area']
        dict_exp = {
            'df_train': train_dataset_raw,
            'df_test': test_dataset_raw,
            'all_features': True,
            'one_features_name': '',
            'blacklist': blacklist_exp
        }

    pearsonr_list = []
    rmse_list = []

    for nt in tqdm(range(1, 11)):
        # Set pipeline path based on the condition
        if only_growth_rate:
            pipeline_path = os.path.join(args.model_save_path, f'pipeline_lasso_only_growth_rate_{nt}.pkl')
        else:
            pipeline_path = os.path.join(args.model_save_path, f'pipeline_lasso_{nt}.pkl')

        pipeline = joblib.load(pipeline_path)

        # Get the processed training and test datasets
        train_dataset, test_dataset = get_datasets(
            all_features=dict_exp['all_features'],
            df_train=dict_exp['df_train'],
            df_test=dict_exp['df_test'],
            one_features_name=dict_exp['one_features_name'],
            blacklist=dict_exp['blacklist'],
            nb_timesteps=nt
        )

        y_train = train_dataset_raw['y_max_extend']
        y_test = test_dataset_raw['y_max_extend']

        # Make predictions
        y_pred = pipeline.predict(test_dataset)

        # Compare predictions to true values
        pearsonr_score = pearsonr(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_pred, y_test))

        pearsonr_list.append(pearsonr_score[0])
        rmse_list.append(rmse)

    # Save metrics
    if only_growth_rate:
        save_metrics(pearsonr_list, args.metrics_save_path, 'pearsonr_list_lasso_only_growth_rate.txt')
        save_metrics(rmse_list, args.metrics_save_path, 'rmse_list_lasso_only_growth_rate.txt')
    else:
        save_metrics(pearsonr_list, args.metrics_save_path, 'pearsonr_list_lasso.txt')
        save_metrics(rmse_list, args.metrics_save_path, 'rmse_list_lasso.txt')
