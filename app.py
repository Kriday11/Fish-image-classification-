import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os
from PIL import Image
# Assuming you have a train_generator defined in fish_classifier.py
# Load model
train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        validation_split=0.2,
        rotation_range=20,
        zoom_range=0.2,
        horizontal_flip=True
    )

train_generator = train_datagen.flow_from_directory(
        'dataset',
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical',
        subset='training'
    )

model = tf.keras.models.load_model("fish_vgg16_model.h5")

# Set class labels (update this based on your training folders)
class_names = list(train_generator.class_indices.keys())  # Save this list during training if needed

# Streamlit UI
st.title("🐟 Fish Image Classifier")
st.write("Upload an image of a fish, and I'll tell you its species!")

uploaded_file = st.file_uploader("Choose a fish image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded Image', use_column_width=True)

    # Preprocess image
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # Prediction
    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions)]
    confidence = np.max(predictions) * 100

    st.write(f"🎯 Prediction: **{predicted_class}**")
    st.write(f"📊 Confidence: **{confidence:.2f}%**")
