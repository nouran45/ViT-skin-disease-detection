# scripts/split_dataset.py

import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split

# Paths
dataset_dir = "datasets/processed_images"
labels_csv = "datasets/HAM10000_metadata.csv"  # already copied
output_dir = "datasets"

# Make sure output folders exist
os.makedirs(os.path.join(output_dir, "train"), exist_ok=True)
os.makedirs(os.path.join(output_dir, "val"), exist_ok=True)
os.makedirs(os.path.join(output_dir, "test"), exist_ok=True)

# Load metadata
metadata = pd.read_csv(labels_csv)

# The 'image_id' column has the image names without extension
# The 'dx' column has the label (disease type)
image_paths = []
labels = []

for index, row in metadata.iterrows():
    img_name = row["image_id"] + ".jpg"
    img_path = os.path.join(dataset_dir, img_name)
    if os.path.exists(img_path):
        image_paths.append(img_path)
        labels.append(row["dx"])

# Split dataset
X_train, X_temp, y_train, y_temp = train_test_split(
    image_paths, labels, test_size=0.30, stratify=labels, random_state=42
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
)

# Function to copy files into folders
def copy_files(X, y, split_name):
    for img_path, label in zip(X, y):
        label_dir = os.path.join(output_dir, split_name, label)
        os.makedirs(label_dir, exist_ok=True)
        shutil.copy(img_path, os.path.join(label_dir, os.path.basename(img_path)))

# Copy files
copy_files(X_train, y_train, "train")
copy_files(X_val, y_val, "val")
copy_files(X_test, y_test, "test")

print("✅ Dataset split successfully into Train, Validation, and Test sets.")
