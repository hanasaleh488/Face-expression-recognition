import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps

# Load model
model = tf.keras.models.load_model("model.keras")

class_names = {0: "Angry", 1: "Disgust", 2: "Fear", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprise"}

# Title
st.title("Facial Expression Recognition")

# File uploader
image = st.file_uploader("Upload an image (JPG/JPEG/PNG)...", type=["jpg", "jpeg", "png"])

# Prediction
if image is not None:
    st.image(image, caption="Uploaded Image", use_column_width=True)
    image = Image.open(image)
    image = ImageOps.grayscale(image)
    image = image.resize([96, 96])
    image = np.array(image) / 255.0

    # Predict
    pred = model.predict(np.expand_dims(image, axis=0))
    pred_label = np.argmax(pred)
    pred_class = class_names[pred_label]

    # Display result
    st.write(f"Predicted Label: **{pred_class}**")