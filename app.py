import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

model = tf.keras.models.load_model("/content/cats_dogs_model.keras")

st.title("🐾 Cat vs Dog Classifier")

uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded Image', use_column_width=True)

    # Preprocess
    img = img.resize((180, 180))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # normalize if model trained on 0-1 range

    predictions = model.predict(img_array)
    score = predictions[0]

    if score < 0.5:
        st.success("It's a **Cat! 🐱**")
    else:
        st.success("It's a **Dog! 🐶**")
