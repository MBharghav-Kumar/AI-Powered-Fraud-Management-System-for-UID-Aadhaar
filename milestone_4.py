import streamlit as st
import numpy as np
import cv2
from PIL import Image
import pandas as pd
from datetime import datetime
import pytesseract
import re

IMG_SIZE = 128

st.set_page_config(page_title="Fraud Detection", layout="centered")

# =========================
# INIT HISTORY
# =========================
if "history" not in st.session_state:
    st.session_state.history = []

# =========================
# SIDEBAR NAVIGATION
# =========================
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Detect Fraud", "View History"])

# =========================
# AADHAAR VALIDATION FUNCTION
# =========================
def is_aadhaar(image):
    try:
        text = pytesseract.image_to_string(image)

        # Aadhaar number pattern
        uid = re.findall(r"\d{4}\s\d{4}\s\d{4}", text)

        # Keywords check
        keywords = ["aadhaar", "uidai", "government of india"]
        found_keyword = any(word in text.lower() for word in keywords)

        if len(uid) > 0 and found_keyword:
            return True
        else:
            return False
    except:
        return False

# =========================
# FRAUD DETECTION FUNCTION
# =========================
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

    if image is not None:
        st.image(image, caption="Input Image", use_column_width=True)

        # STEP 1: Check Aadhaar
        if not is_aadhaar(image):
            result = "NOT AADHAAR ❌"
            st.warning("❌ Uploaded document is NOT Aadhaar")

        else:
            # STEP 2: Fraud Detection
            result = predict_image(image)

            if result == "FRAUD":
                st.error("⚠️ Fraudulent Document Detected!")
            elif result == "GENUINE":
                st.success("✅ Genuine Aadhaar Document")
            elif result == "INVALID":
                st.warning("❌ Invalid Aadhaar Image")
            else:
                st.error("Error processing image")

        st.write("Prediction:", result)

        # SAVE HISTORY
        st.session_state.history.append({
            "Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Result": result
        })

# =========================
# PAGE 2: HISTORY
# =========================
elif page == "View History":

    st.title("📜 Detection History")

    if len(st.session_state.history) > 0:
        df = pd.DataFrame(st.session_state.history)
        st.dataframe(df)
    else:
        st.write("No history available")

    if st.button("Clear History"):
        st.session_state.history = []
        st.success("History Cleared!")
