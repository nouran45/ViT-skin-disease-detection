# scripts/clean_and_resize.py

import os
import cv2

# Set your input and output paths
INPUT_DIR = 'datasets/raw_images'
OUTPUT_DIR = 'datasets/processed_images'

# Define target resize dimensions
TARGET_SIZE = (224, 224)  # 224x224 to match EfficientNet/ResNet inputs

# Create the output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Loop through all images
for filename in os.listdir(INPUT_DIR):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        img_path = os.path.join(INPUT_DIR, filename)
        img = cv2.imread(img_path)

        if img is not None:
            # Resize the image
            img_resized = cv2.resize(img, TARGET_SIZE)
            # Save to output folder
            output_path = os.path.join(OUTPUT_DIR, filename)
            cv2.imwrite(output_path, img_resized)
            print(f"Processed: {filename}")
        else:
            print(f"Failed to load: {filename}")

print("✅ All images resized and saved in:", OUTPUT_DIR)
