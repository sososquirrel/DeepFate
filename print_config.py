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
import config

def print_config(output_folder):
    
    
    with open(os.path.join(output_folder, 'config.txt'), "w") as file:
        file.write("Printing configuration variables:\n")
        file.write("="*50)
            
        for attr in dir(DeepFate.config):
            if not attr.startswith("__"):
                value = getattr(config, attr)
                if isinstance(value, (list, tuple, np.ndarray)):
                    file.write(f"{attr}:\n")
                    for item in value:
                        file.write(f"  - {item} \n")
                elif isinstance(value, dict):
                    file.write(f"{attr}:\n")
                    for key, item in value.items():
                        file.write(f"  - {key}: {item}\n")
                else:
                    file.write(f"{attr}: {value}\n")
    
    
    
if __name__ == '__main__':
    
    # Parse arguments from the user
    parser = argparse.ArgumentParser(description='Arguments training')
    parser.add_argument('--pathfolder', help='pathfolder', type=str, required=True)
    args = parser.parse_args()
    
    print_config(output_folder = args.pathfolder)
