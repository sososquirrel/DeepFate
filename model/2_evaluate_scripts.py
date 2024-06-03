import sys
module_dir = '/home/b/b381993'
sys.path.append(module_dir)
import DeepFate
from DeepFate.model.utils_model import *
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from tqdm import tqdm
import argparse
import pandas as pd

if __name__ == '__main__':
   
    # Parse arguments from the user
    parser = argparse.ArgumentParser(description='Arguments training')
    parser.add_argument('--pathfolder', help='pathfolder', type=str, required=True)
    args = parser.parse_args()


    # Get the folder path from the command line argument
    folder_path = args.pathfolder
    
    folder_path_models = os.path.join(folder_path, 'saved_models')
    #os.makedirs(folder_path_models)
    
    folder_path_stats = os.path.join(folder_path, 'saved_stats')
    os.makedirs(folder_path_stats, exist_ok=True)

    # Search for the .csv file within the folder
    csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

    if not csv_files:
        print("No .csv files found in the specified folder.")
        sys.exit(1)

    # Assuming there is only one .csv file in the folder, you can choose the first one
    csv_file = os.path.join(folder_path, csv_files[0])
    
    #### spliting train and test
    path_train = os.path.join(folder_path, 'train_dataset.csv')
    path_test = os.path.join(folder_path, 'test_dataset.csv')
    df_train = pd.read_csv(path_train)
    df_test = pd.read_csv(path_test)

    ##EVALUATE
    ###only growth rate
    dfs=[]
    for model_str in ['Lasso', 'RandomForest', 'MLPRegressor']:
        pearsonr_list, rmse_list = evaluate_lasso_model(all_features=False,
                                                         model_str=model_str,
                                                         df_train=df_train,
                                                         df_test=df_test,
                                                         output_path_model=f'ONLY_GROWTH_RATE',
                                                         one_features_name='gradient_area',
                                                         folder_path=folder_path_models)
        # Create a DataFrame for the model
        df_model = pd.DataFrame({'model': [f'{model_str}_{i}' for i in range(len(pearsonr_list))],
                                 'pearsonr': pearsonr_list,
                                 'rmse': rmse_list})
        # Append the DataFrame to the list
        dfs.append(df_model)

    # Concatenate all DataFrames into one big DataFrame
    all_models_df = pd.concat(dfs, ignore_index=True)
    
    path_all_models_df = os.path.join(folder_path_stats, 'all_models_only_growth_rate.csv')
    all_models_df.to_csv(path_all_models_df, index=False)


    ### all feaures

    dfs=[]
    for model_str in ['Lasso', 'RandomForest', 'MLPRegressor']:
        dict_model = {}

        pearsonr_list, rmse_list = evaluate_lasso_model(all_features= True,
                            model_str=model_str,
                            df_train = df_train,
                            df_test = df_test,
                            output_path_model= f'ALL_FEATURES',
                            folder_path=folder_path_models)

        
        # Create a DataFrame for the model
        df_model = pd.DataFrame({'model': [f'{model_str}_{i}' for i in range(len(pearsonr_list))],
                                 'pearsonr': pearsonr_list,
                                 'rmse': rmse_list})
        # Append the DataFrame to the list
        dfs.append(df_model)

    # Concatenate all DataFrames into one big DataFrame
    all_models_df = pd.concat(dfs, ignore_index=True)
    
    path_all_models_df = os.path.join(folder_path_stats, 'all_models_all_features.csv')
    all_models_df.to_csv(path_all_models_df, index=False)

