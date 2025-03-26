import h5py
import os

def split_h5_file(input_filename, output_dir, max_chunk_size_gb=8):
    # Open the original .h5 file
    with h5py.File(input_filename, 'r') as f:
        # Get the datasets 'X', 'z', and 'y' (adjust according to your actual file structure)
        X = f['X']
        z = f['z']
        y = f['y']
        
        # Determine the total number of items in 'X' (and z, y)
        total_items = len(X)
        
        # Estimate the size of each item to determine how many items to put in each chunk
        item_size = X.id.get_storage_size() / total_items  # size of one item
        max_items_per_chunk = (max_chunk_size_gb * 1024**3) // item_size
        
        # Ensure the output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Split and save chunks
        chunk_start = 0
        chunk_index = 1
        while chunk_start < total_items:
            chunk_end = min(chunk_start + max_items_per_chunk, total_items)
            
            # Make sure chunk_start and chunk_end are integers
            chunk_start = int(chunk_start)
            chunk_end = int(chunk_end)
            
            # Define output file path
            output_filename = os.path.join(output_dir, f"DEEPFATE_DATASET_chunk_{chunk_index}.h5")
            
            # Create a new HDF5 file to store the chunk
            with h5py.File(output_filename, 'w') as f_out:
                # Copy the 'X', 'z', and 'y' datasets to the new file chunk
                f_out.create_dataset('X', data=X[chunk_start:chunk_end])
                f_out.create_dataset('z', data=z[chunk_start:chunk_end])
                f_out.create_dataset('y', data=y[chunk_start:chunk_end])
            
            print(f"Created chunk {chunk_index} from {chunk_start} to {chunk_end}")
            
            # Move to the next chunk
            chunk_start = chunk_end
            chunk_index += 1

# Usage
input_filename = '/work/bb1153/b381993/data/VERSION_FEBRUARY_2025_3D_v3/DEEPFATE_DATASET.h5'
output_dir = '/work/bb1153/b381993/data/split_DEEPFATE'  # Adjust this path to your desired output directory

split_h5_file(input_filename, output_dir)
