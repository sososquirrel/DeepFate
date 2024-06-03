import sys
module_dir = '/home/b/b381993'
sys.path.append(module_dir)
import DeepFate
from DeepFate.model.utils_model import *
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from tqdm import tqdm
import argparse
import pyarrow
import pandas as pd
import fastparquet

if __name__ == '__main__':
   
    # Parse arguments from the user
    parser = argparse.ArgumentParser(description='Arguments training')
    parser.add_argument('--pathfolder', help='pathfolder', type=str, required=True)
    args = parser.parse_args()


    # Get the folder path from the command line argument
    folder_path = args.pathfolder
    
    
    folder_path_models = os.path.join(folder_path, 'saved_models')

    
    folder_path_preds = os.path.join(folder_path, 'saved_preds')
    os.makedirs(folder_path_preds, exist_ok=True)

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
        y_test_list, y_preds_list = predict_model(all_features= False,
                            model_str=model_str,
                             df_train = df_train,
                             df_test = df_test,
                            output_path_model = f'ONLY_GROWTH_RATE',
                            one_features_name= 'gradient_area',
                            folder_path=folder_path_models)
        
        #print('y_test_list type', all([isinstance(aa,float) for aa in y_test_list]))
        #print('y_preds_list type', all([isinstance(aa,float) for aa in y_preds_list]))
        
        # Create a DataFrame for the model
        df_model = pd.DataFrame({'model': [f'{model_str}_{i}' for i in range(len(y_test_list))],
                                 'truth': y_test_list,
                                 'preds': y_preds_list})
        # Append the DataFrame to the list
        dfs.append(df_model)

    # Concatenate all DataFrames into one big DataFrame
    all_models_df = pd.concat(dfs, ignore_index=True)
    
    path_all_models_df = os.path.join(folder_path_preds, 'preds_all_models_only_growth_rate.parquet')
    all_models_df.to_parquet(path_all_models_df)


    ### all feaures

    dfs=[]
    for model_str in ['Lasso', 'RandomForest', 'MLPRegressor']:
        dict_model = {}

        y_test_list, y_preds_list = predict_model(all_features= False,
                            model_str=model_str,
                             df_train = df_train,
                             df_test = df_test,
                            output_path_model = f'ONLY_GROWTH_RATE',
                            one_features_name= 'gradient_area',
                            folder_path=folder_path_models)


        # Create a DataFrame for the model
        df_model = pd.DataFrame({'model': [f'{model_str}_{i}' for i in range(len(y_preds_list))],
                                 'preds': y_preds_list,
                                 'truth': y_test_list})
        # Append the DataFrame to the list
        dfs.append(df_model)

    # Concatenate all DataFrames into one big DataFrame
    all_models_df = pd.concat(dfs, ignore_index=True)

    path_all_models_df = os.path.join(folder_path_preds, 'preds_all_models_all_features.parquet')
    all_models_df.to_parquet(path_all_models_df)