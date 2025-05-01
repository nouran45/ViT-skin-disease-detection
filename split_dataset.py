# scripts/split_dataset.py

import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split

# === Paths ===
processed_images_dir = "datasets/processed_images"
labels_csv = "datasets/combined_labels.csv"
output_base = "datasets"

# === Ensure output folders exist ===
for split in ['train', 'val', 'test']:
    os.makedirs(os.path.join(output_base, split), exist_ok=True)

# === Load combined metadata ===
df = pd.read_csv(labels_csv)

# Build full image paths
df['path'] = df['image_id'].apply(lambda x: os.path.join(processed_images_dir, x))

# Filter to only existing images
df = df[df['path'].apply(os.path.exists)]

# === 80/10/10 Split ===
# Step 1: 80% train, 20% temp (to be split into val/test)
X_train, X_temp, y_train, y_temp = train_test_split(
    df['path'], df['dx'], test_size=0.20, stratify=df['dx'], random_state=42
)

# Step 2: 10% val, 10% test from temp
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42
)

# === Copy files into folders ===
def copy_to_split(X, y, split_name):
    for path, label in zip(X, y):
        label_dir = os.path.join(output_base, split_name, label)
        os.makedirs(label_dir, exist_ok=True)
        shutil.copy(path, os.path.join(label_dir, os.path.basename(path)))

# Copy each split
copy_to_split(X_train, y_train, "train")
copy_to_split(X_val, y_val, "val")
copy_to_split(X_test, y_test, "test")

# === Final output ===
print("✅ Dataset split complete.")
print("📦 Train:", len(X_train), "| Val:", len(X_val), "| Test:", len(X_test))
