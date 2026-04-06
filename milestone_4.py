import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from PIL import Image

# Load model
model = load_model("fraud_model.h5")

IMG_SIZE = 128

# Title
st.title("AI Powered Aadhaar Fraud Detection System")

st.write("Upload an Aadhaar image to check if it is Genuine or Fraud.")

# Upload file
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

def predict_image(image):
    img = np.array(image)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    img = np.reshape(img, (1, IMG_SIZE, IMG_SIZE, 3))

    pred = model.predict(img)[0][0]

    return "FRAUD" if pred > 0.5 else "GENUINE"

# When user uploads
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    result = predict_image(image)

    # Real-time alert
    if result == "FRAUD":
        st.error("⚠️ Fraudulent Document Detected!")
    else:
        st.success("✅ Genuine Document")

    st.write("Prediction:", result)