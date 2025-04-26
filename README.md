🩺 Skin Disease Fusion Project
A machine learning project inspired by the “An Integrated Deep Learning Model with EfficientNet and ResNet for Accurate Multi-Class Skin Disease Classification” research paper.


📚 Project Overview
This project aims to classify skin diseases using a fusion of EfficientNetB0, EfficientNetB2, and ResNet50 models.
The dataset used is the HAM10000 skin lesion dataset.

🛠 Project Structure
bash
Copy
Edit
skin_disease_fusion_project/
├── datasets/
│   ├── raw_images/              # Original HAM10000 images
│   ├── processed_images/        # Resized images (224x224)
│   ├── train/                   # Training set (70%)
│   ├── val/                     # Validation set (15%)
│   ├── test/                    # Testing set (15%)
│   ├── HAM10000_metadata.csv    # Metadata file
│   └── labels.csv               # Clean labels file
│
├── scripts/
│   ├── clean_and_resize.py      # Script to resize and clean images
│   ├── create_labels.py         # Script to generate labels from metadata
│   ├── split_dataset.py         # Script to split data into train/val/test
│   └── loader_dataset.py        # Script to create TensorFlow data loaders
│
├── venv/                        # Virtual environment (not uploaded)
├── requirements.txt             # Python dependencies
└── README.md                    # Project documentation
🧩 Phase 1: Data Preparation (CPU Member)
✅ Raw images resized and cleaned to 224x224 resolution.

✅ Labels extracted from metadata and saved.

✅ Dataset split into Train (70%), Validation (15%), and Test (15%) sets.

✅ TensorFlow dataset loaders (generators) prepared and verified.

⚡ Phase 2: Model Development (GPU Member)
Build EfficientNetB0, EfficientNetB2, and ResNet50 base models individually.

Apply feature fusion strategy by combining extracted features from all three networks.

Add fully connected layers and output classification head.

Train on GPU with proper callbacks (early stopping, checkpoint saving, learning rate scheduling).
