# scripts/move_and_rename_isic_images.py

print("🟡 Script is running...")

import os
import shutil

# Source: original ISIC images
ISIC_SRC = "datasets/raw_isic_images"

# Destination: where HAM images already are
DEST = "datasets/raw_images"

# Create destination if it doesn't exist
os.makedirs(DEST, exist_ok=True)

print(f"📁 Looking for images in: {ISIC_SRC}")

found_any = False

# Loop through and rename+copy images
for filename in os.listdir(ISIC_SRC):
    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        found_any = True
        src_path = os.path.join(ISIC_SRC, filename)
        new_filename = f"isic_{filename}"
        dest_path = os.path.join(DEST, new_filename)
        shutil.copy(src_path, dest_path)
        print(f"✅ Copied: {filename} → {new_filename}")

if not found_any:
    print("❌ No image files found in", ISIC_SRC)
else:
    print("\n🎉 All ISIC images renamed and copied to:", DEST)
