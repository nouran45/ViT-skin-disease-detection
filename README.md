#  Vision Transformer (ViT) for Skin Disease Classification

This repository contains a fine-tuned **Vision Transformer (ViT)** model for **multi-class skin disease classification**, designed to be part of a scalable, AI-powered medical assistance platform.

---

##  Project Overview

Skin diseases affect millions worldwide, and early diagnosis is often limited by the availability of dermatologists. Our model uses deep learning, specifically ViT, to classify dermoscopic and clinical images across **15 different skin conditions**, achieving **97.4% accuracy**.

 **Model**:  
[🔗Model on Kaggle](https://www.kaggle.com/models/nourshehata183/skin-disease-classification-model-vit)

---

##  Supported Disease Classes (15)

| No. | Class Name                    | Description                                                   |
|-----|-------------------------------|---------------------------------------------------------------|
| 1   | **Enfeksiyonel**              | General label for infectious skin conditions                  |
| 2   | **Ekzama**                    | Inflammatory eczema or dermatitis                             |
| 3   | **Akne**                      | Clinical acne (pimples, blackheads, pustules)                 |
| 4   | **Pigment**                   | Pigmented skin lesions (non-cancerous)                        |
| 5   | **Benign**                    | General benign tumors or skin growths                         |
| 6   | **Malign**                    | General label for malignant or cancerous lesions              |
| 7   | **Acne***                     | Additional acne dataset (separate from Akne)                  |
| 8   | **Actinic Keratosis**         | Pre-cancerous, sun-induced skin damage                        |
| 9   | **Basal Cell Carcinoma (BCC)**| Most common type of skin cancer                               |
| 10  | **Benign Keratosis**          | Non-cancerous keratotic growths (e.g., seborrheic keratosis)  |
| 11  | **Dermatofibroma**            | Benign skin nodules often caused by minor trauma              |
| 12  | **Melanocytic Nevus**         | Common moles                                                  |
| 13  | **Melanoma**                  | Most dangerous type of skin cancer                            |
| 14  | **Vascular Lesion**           | Includes angiomas, hemangiomas, etc.                          |
| 15  | **Warts/Molluscum**           | Viral infections like warts and molluscum contagiosum         |


---

##  Dataset Sources

- [HAM10000](https://doi.org/10.1038/sdata.2018.161)
- [ISIC 2019 Challenge](https://challenge2019.isic-archive.com/)
- [Preprocessed images HAM10000&ISIC2019](https://www.kaggle.com/datasets/nour12347653/skin-disease-detection-dataset-ham10000-isic)
- [Kaggle Clinical Dataset](https://www.kaggle.com/datasets/ascanipek/skin-diseases)
- [Kaggle Acne Dataset](https://www.kaggle.com/datasets/nayanchaure/acne-dataset)

---

##  Preprocessing Pipeline

### 🔹 1. Image Resizing
- All images were resized to **224×224 pixels**.
- Dermoscopic images (originally high-res) were **downsampled**.
- Clinical photos were **cropped** or padded to maintain aspect ratio.

### 🔹 2. Normalization
- Pixel values scaled to `[0, 1]`.
- Normalized using **ImageNet statistics**:
  - Mean: `(0.485, 0.456, 0.406)`
  - Std Dev: `(0.229, 0.224, 0.225)`

### 🔹 3. Augmentation
Used during training to improve generalization:
- Random horizontal/vertical flipping
- Rotation up to ±30°
- Zoom/Crop up to 10%
- Brightness and contrast changes
- Black border padding if needed

### 🔹 4. Class Imbalance Handling
- Used **class-weighted loss**: classes with fewer examples received more weight.
- **Targeted augmentation** applied more frequently to rare classes (e.g. vascular lesions).
- No hard oversampling used to avoid overfitting.

### 🔹 5. Dataset Splitting
- Stratified splitting to ensure class balance:
  - Train: 80%
  - Validation: 10%
  - Test: 10%
- Patient-wise splitting to prevent data leakage (especially for HAM10000)

---

##  Model Architecture

| Feature         | Description                           |
|----------------|---------------------------------------|
| Type            | Vision Transformer (ViT-B/16)         |
| Pretrained on   | ImageNet-21k                          |
| Fine-tuned on   | Combined multi-source dataset         |
| Parameters      | ~85 Million                           |
| Input Resolution| 224 × 224                             |

---

##  Save the Model 

```python
import torch
from transformers import ViTForImageClassification

vit_model = ViTForImageClassification.from_pretrained("nourshehata183/skin-disease-classification-model-vit")
model_save_path = "/kaggle/working/"

# Save weights and config
torch.save(vit_model.state_dict(), f"{model_save_path}/pytorch_model.bin")
vit_model.config.save_pretrained(model_save_path)

# Optional tokenizer
if hasattr(vit_model, 'tokenizer'):
    vit_model.tokenizer.save_pretrained(model_save_path)

print("Model saved successfully.")
```
---


## 📊 Evaluation Metrics

| Metric     | Value  |
|------------|--------|
| **Accuracy**   | 97.4%  |
| **Precision**  | 87%    |
| **Recall**     | 86%    |
| **F1-Score**   | 85%    |

Results show strong generalization across both clinical and dermoscopic images.  

---

##  Deployment

This model is **production-ready** and can be deployed in real-world healthcare settings:

- ✅ **Web and Mobile Health Platforms**
- ✅ **API Integration** with Flask, FastAPI, or Gradio
- ✅ **Confidence-Based Triage System**
  - High-confidence **malignant predictions** trigger **urgent alerts**
  - **Benign outcomes** return self-care suggestions
  - **Low-confidence predictions** indicate poor-quality or ambiguous inputs

---

---

##  Contact
- **Kaggle DataSet**: [ViT Classification Model](https://www.kaggle.com/datasets/nour12347653/skin-disease-detection-dataset-ham10000-isic)

- **Kaggle Model**: [ViT Classification Model](https://www.kaggle.com/models/nourshehata183/skin-disease-classification-model-vit)

---

## 👥 Authors

This project was developed as part of an academic research initiative by:

- **Nouran Salama**  
- **Nour Atef** 
