#!/usr/bin/env python3
import os
import glob
import pandas as pd
import re
import shutil

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, required=True, choices=['DRIVE', 'LESAV', 'CREMI'], help='Dataset to process')
parser.add_argument('--data_root_dir', type=str, required=True, help='Root folder of original dataset (e.g. datasets/DRIVE_dataset)')
parser.add_argument('--output_dir', type=str, default=None, help='Where to save renamed files/CSVs. Default: in-place')
args = parser.parse_args()


DATASET = args.dataset
dataset_dir = args.data_root_dir
output_dir = args.output_dir

# Process each fold
folds = ['fold_1_1', 'fold_1_2', 'fold_2_1', 'fold_2_2']

for fold in folds:
    print(f"\nProcessing {fold}...")
    # Get test files for this fold
    test_files_path = os.path.join(dataset_dir, fold, 'test_files.txt')
    
    # Check if the test files exist
    if not os.path.exists(test_files_path):
        print(f"Warning: Test files path {test_files_path} does not exist. Skipping.")
        continue
    
    # Read the test files
    with open(test_files_path, 'r') as f:
        test_files = f.read().strip().split('\n')
    
    import re

    image_numbers = []
    original_filenames = []

    for file_path in test_files:
        filename = os.path.basename(file_path)
        if DATASET == 'DRIVE':
            # Example DRIVE: 39_training.tif or 21_test.tif or im0319.ppm (you may need to adapt this if your DRIVE filenames are different)
            match = re.search(r'/(\d+)_', file_path)
            if not match:
                match = re.search(r'(\d+)', filename)
            if match:
                image_numbers.append(match.group(1))
                original_filenames.append(filename)
            else:
                print(f"Warning: Could not extract image number from {file_path}")
        elif DATASET == 'LESAV':
            # LESAV: files are like '123.pgm' or '123.png'
            match = re.match(r'(\d+)\.(?:pgm|png)$', filename, re.IGNORECASE)
            if match:
                image_numbers.append(match.group(1))
                original_filenames.append(filename)
            else:
                print(f"Warning: Could not extract image number from {file_path}")
        elif DATASET == 'CREMI':
            # CREMI: A_17.png, B_42.png, C_5.png
            match = re.search(r'([A-Z])_(\d+)\.png$', filename)
            if match:
                image_numbers.append(f"{match.group(1)}_{match.group(2)}")
                original_filenames.append(filename)
            else:
                print(f"Warning: Could not extract image number from {file_path}")
        else:
            raise ValueError(f"Unknown DATASET: {DATASET}")

    
    # Check if we found any image numbers
    if not image_numbers:
        print(f"Warning: No image numbers found in {test_files_path}. Skipping.")
        continue
    
    # Read the CSV file to map indices to original binary file names
    csv_path = os.path.join(output_dir, fold, 'test_metrics.csv')
    
    # Check if the CSV exists
    if not os.path.exists(csv_path):
        print(f"Warning: CSV file {csv_path} not found. Skipping.")
        continue
    
    try:
        # Read the CSV and map the Image column to the image numbers
        df = pd.read_csv(csv_path)
        
        # Check if "Image" column exists
        if 'Image' not in df.columns:
            print(f"Warning: 'Image' column not found in {csv_path}. Looking for alternative columns...")
            image_col = None
            for col in df.columns:
                if 'image' in col.lower() or 'file' in col.lower():
                    image_col = col
                    break
            
            if image_col:
                print(f"Using column '{image_col}' instead.")
            else:
                print(f"No suitable column found in {csv_path}. Assuming sequential order.")
                # Create a sequential mapping
                df['Image'] = list(range(1, len(df) + 1))
                image_col = 'Image'
        else:
            image_col = 'Image'
            
        # Create a backup of the CSV file
        csv_backup = csv_path + '.backup'
        if not os.path.exists(csv_backup):
            print(f"Creating backup of CSV file to {csv_backup}")
            shutil.copy2(csv_path, csv_backup)
        
        # Path to the binary folder
        binary_folder = os.path.join(output_dir, fold, 'binary')
        if not os.path.exists(binary_folder):
            print(f"Warning: Binary folder {binary_folder} not found. Skipping.")
            continue
            
        # Create a backup of the binary folder
        backup_folder = os.path.join(output_dir, fold, 'binary_backup')
        if not os.path.exists(backup_folder):
            print(f"Creating backup of binary folder to {backup_folder}")
            shutil.copytree(binary_folder, backup_folder)
        
        # Find all binary files in the binary folder
        binary_files = glob.glob(os.path.join(binary_folder, 'binary_*.png'))
        
        # If no binary files with 'binary_*.png' found, try without .png extension
        if not binary_files:
            binary_files = glob.glob(os.path.join(binary_folder, 'binary_*'))
            
        # If still no files, look for any files
        if not binary_files:
            binary_files = glob.glob(os.path.join(binary_folder, '*'))
            if binary_files:
                print(f"Found {len(binary_files)} files without 'binary_' prefix in {binary_folder}")
        
        if not binary_files:
            print(f"Warning: No binary files found in {binary_folder}. Skipping.")
            continue
            
        # Extract the indices from filenames (e.g., binary_1.png -> 1)
        file_indices = {}
        for file_path in binary_files:
            filename = os.path.basename(file_path)
            match = re.search(r'binary_(\d+)', filename)
            if match:
                idx = int(match.group(1))
                file_indices[idx] = file_path
            else:
                # For files without the expected format, try to infer the index
                match = re.search(r'(\d+)', filename)
                if match:
                    idx = int(match.group(1))
                    file_indices[idx] = file_path
        
        # Create a mapping from old indices to new image numbers
        index_to_img_num = {}
        for i, img_num in enumerate(image_numbers):
            if i < len(df):
                try:
                    old_idx = int(df[image_col].iloc[i])
                    index_to_img_num[old_idx] = img_num
                except:
                    print(f"Warning: Could not convert value '{df[image_col].iloc[i]}' in row {i} to int. Skipping.")
        
        # Rename the binary files
        for old_idx, img_num in index_to_img_num.items():
            if old_idx in file_indices:
                src_file = file_indices[old_idx]
                # Ensure .png extension
                dst_file = os.path.join(binary_folder, f'binary_{img_num}.png')
                
                if os.path.exists(src_file):
                    print(f"Renaming {src_file} to {dst_file}")
                    # In case the destination already exists
                    if os.path.exists(dst_file) and src_file != dst_file:
                        os.remove(dst_file)
                    os.rename(src_file, dst_file)
                else:
                    print(f"Warning: Source file {src_file} not found.")
            else:
                print(f"Warning: No file found for index {old_idx} in fold {fold}.")
        
        # Update the CSV file with the new image numbers
        df_copy = df.copy()
        for i, row in df.iterrows():
            try:
                old_idx = int(row[image_col])
                if old_idx in index_to_img_num:
                    df_copy.at[i, image_col] = index_to_img_num[old_idx]
                    # Also update the original filename if we have a column for it
                    file_idx = image_numbers.index(index_to_img_num[old_idx])
                    if file_idx < len(original_filenames):
                        for col in df.columns:
                            if 'file' in col.lower() or 'name' in col.lower():
                                df_copy.at[i, col] = original_filenames[file_idx]
            except:
                print(f"Warning: Could not update row {i} in CSV. Skipping.")
        
        # Write the updated CSV
        df_copy.to_csv(csv_path, index=False)
        print(f"Updated CSV file with new image numbers in {csv_path}")
        
        print(f"Successfully processed {fold}")
    
    except Exception as e:
        print(f"Error processing {fold}: {str(e)}")

print("Done renaming files.")