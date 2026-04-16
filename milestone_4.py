import streamlit as st
import numpy as np
import cv2
from PIL import Image

IMG_SIZE = 128

st.title("AI Powered Aadhaar Fraud Detection System")
st.write("Upload an Aadhaar image to check if it is Genuine or Fraud.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

def predict_image(image):
    try:
        img = np.array(image)

        # Debug
        st.write("Image Shape:", img.shape)

        if img is None or img.size == 0:
            return "ERROR: Invalid Image"

        # Convert grayscale if needed
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        img = cv2.resize(img, (128,128))
        img = img / 255.0

        gray = cv2.cvtColor((img*255).astype("uint8"), cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)

        edge_density = np.sum(edges) / (128*128)

        # Debug
        st.write("Edge Density:", edge_density)

        if edge_density < 5:
            return "INVALID DOCUMENT ❌"

        mean_pixel = np.mean(img)

        if mean_pixel < 0.5:
            return "FRAUD ⚠️"
        else:
            return "GENUINE ✅"

    except Exception as e:
        return f"ERROR: {str(e)}"


# MAIN UI
if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file)

        st.image(image, caption="Uploaded Image", use_column_width=True)

        result = predict_image(image)

        # Show result ALWAYS
        if "FRAUD" in result:
            st.error("⚠️ Fraudulent Document Detected!")
        elif "GENUINE" in result:
            st.success("✅ Genuine Document")
        elif "INVALID" in result:
            st.warning("❌ Invalid Document (Not Aadhaar)")
        else:
            st.error(result)

        st.write("Prediction:", result)

    except Exception as e:
        st.error(f"App Error: {str(e)}")
