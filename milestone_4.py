import streamlit as st
import numpy as np
import cv2
from PIL import Image

IMG_SIZE = 128

st.set_page_config(page_title="Fraud Detection", layout="centered")

st.title("AI Powered Aadhaar Fraud Detection System")
st.write("Upload or capture an Aadhaar image to check if it is Genuine or Fraud.")

# 🔹 OPTION SELECTOR
option = st.radio("Choose Input Method:", ["Upload Image", "Capture Image"])

# 🔹 INPUT HANDLING
image = None

if option == "Upload Image":
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)

elif option == "Capture Image":
    captured_image = st.camera_input("Capture Aadhaar Image")
    if captured_image is not None:
        image = Image.open(captured_image)

# 🔹 PREDICTION FUNCTION
def predict_image(image):
    try:
        img = np.array(image)

        if img is None or img.size == 0:
            return "ERROR: Invalid Image"

        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = img / 255.0

        gray = cv2.cvtColor((img * 255).astype("uint8"), cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)

        edge_density = np.sum(edges) / (IMG_SIZE * IMG_SIZE)
        mean_pixel = np.mean(img)

        if edge_density < 2:
            return "INVALID DOCUMENT ❌"
        elif mean_pixel < 0.5:
            return "FRAUD ⚠️"
        else:
            return "GENUINE ✅"

    except Exception as e:
        return f"ERROR: {str(e)}"


# 🔹 MAIN OUTPUT
if image is not None:
    st.image(image, caption="Input Image", use_column_width=True)

    result = predict_image(image)

    if "FRAUD" in result:
        st.error("⚠️ Fraudulent Document Detected!")
    elif "GENUINE" in result:
        st.success("✅ Genuine Document")
    elif "INVALID" in result:
        st.warning("❌ Invalid Document (Not Aadhaar)")
    else:
        st.error(result)

    st.write("Prediction:", result)
