import argparse
import os
import pandas as pd
import shutil

parser = argparse.ArgumentParser()
parser.add_argument('--root_dir', type=str, required=True, help='Directory containing all fold subdirectories')
parser.add_argument('--output_dir', type=str, required=True, help='Where to save merged CSV/images')
args = parser.parse_args()

root_dir = args.root_dir
output_dir = args.output_dir

merged_csv_path = os.path.join(output_dir, "merged_test_metrics.csv")
merged_binary_dir = os.path.join(output_dir, "merged_binary")
os.makedirs(merged_binary_dir, exist_ok=True)

df_list = []

# Iterate through all fold directories (from root_dir!)
for fold in os.listdir(root_dir):
    fold_path = os.path.join(root_dir, fold)
    if os.path.isdir(fold_path):
        csv_file = os.path.join(fold_path, "test_metrics.csv")
        binary_dir = os.path.join(fold_path, "binary")

        # Merge CSV files
        if os.path.exists(csv_file):
            df = pd.read_csv(csv_file)
            df["Fold"] = fold
            df_list.append(df)

        # Move binary images
        if os.path.exists(binary_dir):
            for img_file in os.listdir(binary_dir):
                src_path = os.path.join(binary_dir, img_file)
                dst_path = os.path.join(merged_binary_dir, img_file)
                if os.path.exists(dst_path):
                    base, ext = os.path.splitext(img_file)
                    count = 1
                    while os.path.exists(dst_path):
                        new_name = f"{base}_{count}{ext}"
                        dst_path = os.path.join(merged_binary_dir, new_name)
                        count += 1
                shutil.copy(src_path, dst_path)

# Merge all CSV data and sort by 'Image name'
if df_list:
    merged_df = pd.concat(df_list, ignore_index=True)
    merged_df = merged_df.sort_values(by=["Image"])  # Adjust column name if needed
    merged_df.to_csv(merged_csv_path, index=False)

print(f"✅ Merged CSV saved at: {merged_csv_path}")
print(f"✅ All binary images moved to: {merged_binary_dir}")
