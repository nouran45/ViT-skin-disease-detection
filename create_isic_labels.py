# scripts/create_isic_labels.py

import pandas as pd

# Load CSVs
metadata = pd.read_csv("datasets/ISIC_2019_Training_Metadata.csv")
ground_truth = pd.read_csv("datasets/ISIC_2019_Training_GroundTruth.csv")

# Merge both datasets on 'image' column
df = pd.merge(metadata, ground_truth, on="image")

# Mapping from one-hot labels to unified 10-class labels
label_map = {
    'MEL': 'melanoma',
    'NV': 'melanocytic nevi',
    'BCC': 'basal cell carcinoma',
    'BKL': 'benign keratosis',
    'AK': 'benign keratosis',  # AK sometimes grouped into BKL
    'DF': 'seborrheic keratosis',
    'VASC': 'warts/molluscum',
    'SCC': 'basal cell carcinoma',  # SCC usually handled like BCC
    'UNK': 'eczema',  # fallback for unknown — optional
    'ECZ': 'eczema',
    'PSO': 'psoriasis',
    'TINEA': 'tinea/candidiasis',
    'LICHEN': 'psoriasis',
}

# Extract the one-hot encoded label columns
label_cols = ['MEL', 'NV', 'BCC', 'AK', 'BKL', 'DF', 'VASC', 'SCC']

# Create a column 'dx' from the one-hot labels
df['dx'] = df[label_cols].idxmax(axis=1)

# Map to unified label
df['dx'] = df['dx'].map(label_map)

# Add 'isic_' prefix to match renamed image filenames
df['image_id'] = df['image'].apply(lambda x: f"isic_{x}.jpg")

# Save only needed columns
df[['image_id', 'dx']].to_csv("datasets/isic_labels.csv", index=False)

print("✅ isic_labels.csv created with unified class labels.")
