import streamlit as st
import os
import numpy as np
import tensorflow as tf
from PIL import Image

MODEL_PATH = os.path.join(os.path.dirname(__file__), "best_age_regression_model.keras")
model = tf.keras.models.load_model(MODEL_PATH)


groups = [
    "0: Child (0-12)",
    "1: Teenager (13-19)",
    "2: Young Adult (20-39)",
    "3: Middle-aged (40-59)",
    "4: Senior (60+)",
]
 
st.title("Age Group Prediction App")
st.write("Upload a face image to predict the age group")
 
uploaded_file = st.file_uploader("Upload a face image", type=["jpg", "jpeg", "png"])
 
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", width=300)
 
    if st.button("Predict"):
        image = image.resize((48, 48))
        image_array = np.array(image, dtype="float32") / 255.0
        image_array = np.expand_dims(image_array, axis=0)
 
        probs = model.predict(image_array, verbose=0)[0]
        result = np.argmax(probs)
 
        st.success("Predicted: " + groups[result])
 
        st.write("Probability for each class:")
        st.bar_chart(dict(zip(groups, probs.tolist())))


