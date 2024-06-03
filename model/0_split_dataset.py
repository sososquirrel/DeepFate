import sys
module_dir = '/home/b/b381993'
sys.path.append(module_dir)
import DeepFate
from DeepFate.model.utils_model import *
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from tqdm import tqdm
import argparse
import os
import numpy as np
import pandas as pd

np.random.seed(42)

def split_dataset(output_folder):
    
    csv_file = os.path.join(output_folder, 'interpretable_features_data.csv')
    
    df = pd.read_csv(csv_file)
    print('BEFORE DROPNA', len(df))
    df = df.dropna(axis=1, how='any')
    df = df.dropna(axis=0, how='any')
    print('AFTER DROPNA', len(df))
    random_index = np.random.permutation(np.arange(len(df)))
    N = len(random_index)
    N_train =int(0.75*N) 
    df_train = df.iloc[random_index[:N_train]].reset_index(drop=True)
    df_test = df.iloc[random_index[N_train:]].reset_index(drop=True)
    
    path_train = os.path.join(output_folder, 'train_dataset.csv')
    path_test = os.path.join(output_folder, 'test_dataset.csv')
    
    df_train.to_csv(path_train)
    df_test.to_csv(path_test)

    
if __name__ == '__main__':
    
    # Parse arguments from the user
    parser = argparse.ArgumentParser(description='Arguments training')
    parser.add_argument('--pathfolder', help='pathfolder', type=str, required=True)
    args = parser.parse_args()
    
    split_dataset( output_folder = args.pathfolder)
