import streamlit as st
import numpy as np
import cv2
from PIL import Image
import pandas as pd
from datetime import datetime
import pytesseract
import re

IMG_SIZE = 128

# Load face detector
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

st.set_page_config(page_title="Aadhaar Fraud Detection", layout="centered")

# =========================
# SESSION STATE
# =========================
if "history" not in st.session_state:
    st.session_state.history = []

# =========================
# SIDEBAR
# =========================
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Detect", "History"])

# =========================
# OCR + Aadhaar Validation
# =========================
def is_aadhaar(image):
    try:
        text = pytesseract.image_to_string(image)

        uid = re.findall(r"\d{4}\s\d{4}\s\d{4}", text)

        keywords = ["aadhaar", "uidai", "government of india"]
        keyword_score = sum(word in text.lower() for word in keywords)

        return len(uid) > 0 or keyword_score >= 1
    except:
        return False

# =========================
# FACE DETECTION
# =========================
def detect_face(image):
    img = np.array(image)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    return len(faces) > 0

# =========================
# LOGO DETECTION (simple color check)
# =========================
def detect_logo(image):
    img = np.array(image)

    # Aadhaar logo has orange + green tones
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    lower_orange = np.array([5, 100, 100])
    upper_orange = np.array([20, 255, 255])

    mask = cv2.inRange(hsv, lower_orange, upper_orange)

    return np.sum(mask) > 500

# =========================
# FRAUD DETECTION (lightweight CNN alternative)
# =========================
def detect_fraud(image):
    img = np.array(image)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0

    gray = cv2.cvtColor((img * 255).astype("uint8"), cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)

    edge_density = np.sum(edges) / (IMG_SIZE * IMG_SIZE)
    mean_pixel = np.mean(img)

    if edge_density < 2:
        return "INVALID"
    elif mean_pixel < 0.4:
        return "FRAUD"
    else:
        return "GENUINE"

# =========================
# REPORT DOWNLOAD
# =========================
def generate_report(result):
    df = pd.DataFrame([result])
    return df.to_csv(index=False).encode('utf-8')

# =========================
# PAGE: DETECT
# =========================
if page == "Detect":

    st.title("🪪 Aadhaar Fraud Detection System")

    option = st.radio("Input Method:", ["Upload", "Camera"])

    image = None

    if option == "Upload":
        file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])
        if file:
            image = Image.open(file)

    else:
        file = st.camera_input("Capture Image")
        if file:
            image = Image.open(file)

    if image:
        st.image(image, caption="Input", use_column_width=True)

        # STEP 1: Aadhaar check
        aadhaar_flag = is_aadhaar(image)

        # STEP 2: Face check
        face_flag = detect_face(image)

        # STEP 3: Logo check
        logo_flag = detect_logo(image)

        st.write("🔍 Aadhaar Text Check:", aadhaar_flag)
        st.write("🙂 Face Detected:", face_flag)
        st.write("🎨 Logo Detected:", logo_flag)

        # FINAL DECISION
        if not aadhaar_flag:
            result = "NOT AADHAAR ❌"
            st.error(result)

        elif not face_flag:
            result = "NO FACE FOUND ❌"
            st.warning(result)

        else:
            result = detect_fraud(image)

            if result == "FRAUD":
                st.error("⚠️ Fraud Detected")
            elif result == "GENUINE":
                st.success("✅ Genuine Aadhaar")
            else:
                st.warning("❌ Invalid Image")

        st.write("Final Result:", result)

        # Save history
        record = {
            "Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Result": result
        }
        st.session_state.history.append(record)

        # Download report
        st.download_button(
            label="Download Report",
            data=generate_report(record),
            file_name="report.csv",
            mime="text/csv"
        )

# =========================
# PAGE: HISTORY
# =========================
elif page == "History":

    st.title("📜 History")

    if st.session_state.history:
        df = pd.DataFrame(st.session_state.history)
        st.dataframe(df)
    else:
        st.write("No history")

    if st.button("Clear History"):
        st.session_state.history = []
        st.success("Cleared")
