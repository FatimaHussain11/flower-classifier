import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Load the trained model
model = tf.keras.models.load_model("flower_model.h5")

# Class names (same order as in training)
class_names = ['dandelion', 'daisy', 'tulips', 'sunflowers', 'roses']

# Image preprocessing function
def preprocess_image(image):
    image = image.resize((224, 224))
    image = np.array(image) / 255.0  # Normalize
    image = np.expand_dims(image, axis=0)
    return image

# Streamlit UI
st.title("🌸 Flower Classifier")
st.write("Upload an image of a flower, and I will predict its class.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess and predict
    img = preprocess_image(image)
    prediction = model.predict(img)[0]
    
    # Output
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction)
# trigger rebuild

    st.markdown(f"### 🌼 Prediction: **{predicted_class}**")
    st.markdown(f"**Confidence:** {confidence:.2f}")

