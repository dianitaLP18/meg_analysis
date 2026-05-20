import os
import h5py
import pandas as pd
import numpy as np

# Map the prefixes to numeric categories
LABEL_MAP = {
    'rest': 0,
    'task_motor': 1,
    'task_story_math': 2,
    'task_working_memory': 3
}

def preprocess_matrix(matrix, downsample_factor=4):
    """Applies downsampling and time-wise Z-score normalization."""
    # Downsample columns (time-steps)
    matrix_downsampled = matrix[:, ::downsample_factor]
    
    # Time-wise Z-score normalization
    mean = np.mean(matrix_downsampled, axis=1, keepdims=True)
    std = np.std(matrix_downsampled, axis=1, keepdims=True)
    std[std == 0] = 1.0  # Avoid zero-division
    
    return (matrix_downsampled - mean) / std

def load_dataset_from_folder(folder_path):
    """
    Scans a folder for .h5 files, extracts labels from names,
    and returns arrays of preprocessed data (X) and labels (y).
    """
    X_list = []
    y_list = []
    
    if not os.path.exists(folder_path):
        print(f"Error: Path {folder_path} does not exist.")
        return None, None

    # Get all .h5 files in alphabetic/logical order
    files = sorted([f for f in os.listdir(folder_path) if f.endswith('.h5')])
    
    print(f"Processing {len(files)} files found in: {folder_path}...")
    
    for filename in files:
        # Determine label matching the file prefix string
        label = None
        for prefix, value in LABEL_MAP.items():
            if filename.startswith(prefix):
                label = value
                break
                
        if label is None:
            continue # Skip files that don't match the standard naming conventions
            
        file_fullpath = os.path.join(folder_path, filename)
        
        # Open and load the dataset contents safely
        try:
            with h5py.File(file_fullpath, 'r') as f:
                key = list(f.keys())[0]
                raw_matrix = f[key][()]
                
                # Apply downsampling and normalization
                processed_matrix = preprocess_matrix(raw_matrix, downsample_factor=4)
                
                X_list.append(processed_matrix)
                y_list.append(label)
        except Exception as e:
            print(f"Failed to read {filename}: {e}")

    # Convert structural lists to stable NumPy arrays
    X = np.array(X_list)
    y = np.array(y_list)
    return X, y

# Setting up for cross subject classification
print("Loading cross subject dataset")
X_cross_train, y_cross_train = load_dataset_from_folder("assign2/Final_Project_data/Cross/train")
X_cross_test1, y_cross_test1 = load_dataset_from_folder("assign2/Final_Project_data/Cross/test1")
X_cross_test2, y_cross_test2 = load_dataset_from_folder("assign2/Final_Project_data/Cross/test2")
X_cross_test3, y_cross_test3 = load_dataset_from_folder("assign2/Final_Project_data/Cross/test3")

print(f"\nFinal Train Matrix Shapes: X={X_cross_train.shape}, y={y_cross_train.shape}")
print(f"Final Test1 Matrix Shapes: X={X_cross_test1.shape}, y={y_cross_test1.shape}")
print(f"Final Test2 Matrix Shapes: X={X_cross_test1.shape}, y={y_cross_test2.shape}")
print(f"Final Test3 Matrix Shapes: X={X_cross_test1.shape}, y={y_cross_test3.shape}")


# print("Loading intra subject dataset")
X_intra_train, y_intra_train = load_dataset_from_folder("assign2/Final_Project_data/Intra/train")
X_intra_test, y_intra_test = load_dataset_from_folder("assign2/Final_Project_data/Intra/test")

print(f"\nFinal Intra train Matrix Shapes: X={X_intra_train.shape}, y={y_intra_train.shape}")
print(f"Final Intra test Matrix Shapes: X={X_intra_test.shape}, y={y_intra_test.shape}")
