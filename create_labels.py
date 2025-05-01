# scripts/create_labels.py

import pandas as pd
import os

# Paths
metadata_csv_path = 'datasets/HAM10000_metadata.csv'
output_labels_csv_path = 'datasets/labels.csv'

# Load the metadata
df = pd.read_csv(metadata_csv_path)

# We need only two columns: image_id and dx (diagnosis)
labels_df = df[['image_id', 'dx']]

# Save the labels to a new CSV file
labels_df.to_csv(output_labels_csv_path, index=False)

print("✅ Labels CSV created successfully!")
