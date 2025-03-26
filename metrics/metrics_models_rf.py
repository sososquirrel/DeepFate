from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import sys
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
from tqdm import tqdm
import sys
module_dir = '/home/b/b381993'
sys.path.append(module_dir)
from DeepFate.important_features.utils_important_features import get_datasets


# Dataset
train_dataset_raw = pd.read_csv('/work/bb1153/b381993/data/VERSION_MARCH_NEW_INTERPRETABLE/train_dataset.csv')
test_dataset_raw = pd.read_csv('/work/bb1153/b381993/data/VERSION_MARCH_NEW_INTERPRETABLE/test_dataset.csv')

print(len(train_dataset_raw), len(test_dataset_raw))

only_growth_rate = True

# Define the dictionary for feature selection
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

# Iterate over a range of timesteps for evaluation
for nt in tqdm(range(1, 11)):

    # Define the pipeline path for Random Forest
    if only_growth_rate:
        pipeline_path = f'/home/b/b381993/DeepFate/models/pipeline_rf_only_growth_rate_{nt}.pkl'
    else:
        pipeline_path = f'/home/b/b381993/DeepFate/models/pipeline_rf_{nt}.pkl'
    
    # Load the pipeline
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

    # Define the target variables
    y_train = train_dataset_raw['y_max_extend']
    y_test = test_dataset_raw['y_max_extend']

    # Make predictions using the pipeline
    y_pred = pipeline.predict(test_dataset)

    # Compare predictions to true values
    peasonr_score = pearsonr(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_pred, y_test))

    # Append the results to the lists
    pearsonr_list.append(peasonr_score[0])
    rmse_list.append(rmse)

# Save the Pearson correlation and RMSE results to text files
if only_growth_rate:
    np.savetxt('pearsonr_list_rf_only_growth_rate.txt', pearsonr_list, delimiter=',')
    np.savetxt('rmse_list_rf_only_growth_rate.txt', rmse_list, delimiter=',')
else:
    np.savetxt('pearsonr_list_rf.txt', pearsonr_list, delimiter=',')
    np.savetxt('rmse_list_rf.txt', rmse_list, delimiter=',')
