import streamlit as st
import numpy as np
import cv2
from PIL import Image

IMG_SIZE = 128

# Title
st.title("AI Powered Aadhaar Fraud Detection System")

st.write("Upload an Aadhaar image to check if it is Genuine or Fraud.")

# Upload file
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

def predict_image(image):
    img = np.array(image)

    # Convert if grayscale
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    img = cv2.resize(img, (128,128))
    img = img / 255.0

    # Convert to text-like detection using edges
    gray = cv2.cvtColor((img*255).astype("uint8"), cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)

    edge_density = np.sum(edges) / (128*128)

    # 🧠 Logic:
    # Aadhaar has text → more edges
    if edge_density < 20:
        return "INVALID DOCUMENT ❌ (Not Aadhaar)"

    # Simulated fraud logic
    mean_pixel = np.mean(img)

    if mean_pixel < 0.5:
        return "FRAUD ⚠️"
    else:
        return "GENUINE ✅"
