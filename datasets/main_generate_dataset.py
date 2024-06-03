import sys
import os
from datetime import datetime
import argparse

# Add module directory to the path
module_dir = '/home/b/b381993'
sys.path.append(module_dir)

# Import the required module from DeepFate
import DeepFate
from DeepFate.datasets.generate_precomputed_datasets import precompute_all_mcs

# Parse arguments from the user
parser = argparse.ArgumentParser(description='Arguments generating .h5 dataset')
parser.add_argument('--pathfolder', help='pathfolder', type=str, required=True)
parser.add_argument('--start_index', help='start', type=int, required=True)
parser.add_argument('--stop_index', help='stop', type=int, required=True)
args = parser.parse_args()

if __name__ == '__main__':

    pathfolder = args.pathfolder
    now = datetime.now()
    str_date = now.strftime("%d-%m-%Y-%H%M")
    NAME_DATASET = f'DEEPFATE_{str_date}_index_from_{args.start_index}_to_{args.stop_index}.h5'

    print(NAME_DATASET, 'is about to be generated')

    path_output_h5 = os.path.join(pathfolder, NAME_DATASET)
    precompute_all_mcs(start_index = args.start_index, stop_index = args.stop_index, path_output_h5_file=path_output_h5)

    print(NAME_DATASET, 'has been successfully generated')
    
    # Print the path to the generated .h5 file
    print(f"Generated dataset path: {path_output_h5}")
