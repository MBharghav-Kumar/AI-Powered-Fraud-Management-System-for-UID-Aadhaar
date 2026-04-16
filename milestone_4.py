import streamlit as st
import numpy as np
import cv2
from PIL import Image
import pandas as pd
from datetime import datetime

IMG_SIZE = 128

st.set_page_config(page_title="Fraud Detection", layout="centered")

# 🔹 Initialize history
if "history" not in st.session_state:
    st.session_state.history = []

# 🔹 SIDEBAR NAVIGATION
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Detect Fraud", "View History"])

# =========================
# PAGE 1: DETECT FRAUD
# =========================
if page == "Detect Fraud":

    st.title("AI Powered Aadhaar Fraud Detection System")

    option = st.radio("Choose Input Method:", ["Upload Image", "Capture Image"])

    image = None

    if option == "Upload Image":
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
        if uploaded_file is not None:
            image = Image.open(uploaded_file)

    elif option == "Capture Image":
        captured_image = st.camera_input("Capture Aadhaar Image")
        if captured_image is not None:
            image = Image.open(captured_image)

    def predict_image(image):
        try:
            img = np.array(image)

            if img is None or img.size == 0:
                return "ERROR"

            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            img = img / 255.0

            gray = cv2.cvtColor((img * 255).astype("uint8"), cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, 50, 150)

            edge_density = np.sum(edges) / (IMG_SIZE * IMG_SIZE)
            mean_pixel = np.mean(img)

            if edge_density < 2:
                return "INVALID"
            elif mean_pixel < 0.5:
                return "FRAUD"
            else:
                return "GENUINE"

        except:
            return "ERROR"

    if image is not None:
        st.image(image, caption="Input Image", use_column_width=True)

        result = predict_image(image)

        if result == "FRAUD":
            st.error("⚠️ Fraudulent Document Detected!")
        elif result == "GENUINE":
            st.success("✅ Genuine Document")
        elif result == "INVALID":
            st.warning("❌ Invalid Document")
        else:
            st.error("Error processing image")

        # Save history
        st.session_state.history.append({
            "Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Result": result
        })

# =========================
# PAGE 2: VIEW HISTORY
# =========================
elif page == "View History":

    st.title("📜 Detection History")

    if len(st.session_state.history) > 0:
        df = pd.DataFrame(st.session_state.history)
        st.dataframe(df)
    else:
        st.write("No history available")

    # Clear history button
    if st.button("Clear History"):
        st.session_state.history = []
        st.success("History Cleared!")
