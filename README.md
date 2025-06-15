# 🐟 Multiclass Fish Image Classification

This project classifies images of fish into multiple species using Deep Learning. It includes model training (CNN and Transfer Learning), evaluation, and deployment via a Streamlit web app.

---

## 🚀 Project Overview

- **Domain**: Image Classification  
- **Objective**: Classify fish species using Convolutional Neural Networks (CNN) and Transfer Learning with VGG16  
- **Deployment**: Streamlit Web App  
- **Skills Used**: Python, TensorFlow/Keras, Data Augmentation, Transfer Learning, Streamlit, Model Evaluation

---

## 📁 Folder Structure



---

## 🧠 Model Details

### ✅ CNN from Scratch
- Trained a basic CNN model with 3 Conv layers and dropout

### ✅ VGG16 (Transfer Learning)
- Used pretrained VGG16 as a base
- Added custom Dense layers for classification
- Achieved better performance with reduced training time

---

## 🏋️‍♂️ Training Approach

- Resized images to 224x224
- Normalized pixel values to [0, 1]
- Applied augmentation: rotation, zoom, flip
- Used `ImageDataGenerator` with a 80/20 train-validation split
- Fine-tuned with Adam optimizer and categorical crossentropy

---

## 📊 Model Evaluation

- Metrics used: **Accuracy**, **Loss**, **Validation Accuracy**
- Saved the best model as `fish_vgg16_model.h5`
- Visualized accuracy/loss curves using `matplotlib`

---
## 🔗 External Files

Due to GitHub's upload size limits, large files are hosted externally:

- 📦 Trained Model (`fish_vgg16_model.h5`): [Download here](https://drive.google.com/drive/folders/1kxXMw-2PZX3982aODDyQCnhWV8nE2o0B?usp=drive_link)
- 🐟 Dataset (`Dataset.zip`): [Download here](https://drive.google.com/drive/folders/1kxXMw-2PZX3982aODDyQCnhWV8nE2o0B?usp=drive_link)

After downloading:
- Place `fish_vgg16_model.h5` in the project root folder.
- Unzip `Dataset.zip` and place the folder as `dataset/` in the project root.


## 🌐 Streamlit App

The app allows users to:
- Upload an image of a fish
- Predict the fish species
- Display confidence scores

---

## ▶️ How to Run the App

1. Clone or download the repo
2. Install dependencies:

```bash
pip install -r requirements.txt
