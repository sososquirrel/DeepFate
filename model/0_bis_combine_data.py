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

def combine_dataset(output_folder):
    
    csv_file_train_1 = os.path.join(output_folder, 'train_dataset_1.csv')
    csv_file_train_2 = os.path.join(output_folder, 'train_dataset_2.csv')
    
    csv_file_test_1 = os.path.join(output_folder, 'test_dataset_1.csv')
    csv_file_test_2 = os.path.join(output_folder, 'test_dataset_2.csv')
    
    df_train_1 = pd.read_csv(csv_file_train_1)
    df_train_2 = pd.read_csv(csv_file_train_2)
    
    df_test_1 = pd.read_csv(csv_file_test_1)
    df_test_2 = pd.read_csv(csv_file_test_2)
    
    df_train = pd.concat([df_train_1, df_train_2], axis=0)
    df_test = pd.concat([df_test_1, df_test_2], axis=0)
    
    df_train = df_train.dropna(axis=1, how='any')
    df_train = df_train.dropna(axis=0, how='any')
    
    df_test = df_test.dropna(axis=1, how='any')
    df_test = df_test.dropna(axis=0, how='any')
    
    df_train.to_csv(os.path.join(output_folder, 'train_dataset.csv'), index=False)
    df_test.to_csv(os.path.join(output_folder, 'test_dataset.csv'), index=False)
                   
if __name__ == '__main__':
    
    # Parse arguments from the user
    parser = argparse.ArgumentParser(description='Arguments training')
    parser.add_argument('--pathfolder', help='pathfolder', type=str, required=True)
    args = parser.parse_args()
    
    combine_dataset( output_folder = args.pathfolder)
