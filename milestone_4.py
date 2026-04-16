import streamlit as st
import numpy as np
import cv2
from PIL import Image
import pandas as pd
from datetime import datetime
import pytesseract
import re

IMG_SIZE = 128

# Use built-in Haar cascade (NO FILE NEEDED)
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

st.set_page_config(page_title="Aadhaar Fraud Detection", layout="centered")

# =========================
# SESSION STATE (History)
# =========================
if "history" not in st.session_state:
    st.session_state.history = []

# =========================
# SIDEBAR NAVIGATION
# =========================
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Detect", "History"])

# =========================
# OCR PREPROCESSING
# =========================
def preprocess_for_ocr(image):
    img = np.array(image)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray = cv2.equalizeHist(gray)
    gray = cv2.GaussianBlur(gray, (3,3), 0)
    return gray

# =========================
# AADHAAR TEXT CHECK (IMPROVED)
# =========================
def is_aadhaar(image):
    try:
        processed = preprocess_for_ocr(image)
        text = pytesseract.image_to_string(processed)

        # DEBUG (optional)
        st.write("🔍 OCR Text:", text)

        # Flexible UID patterns
        uid_patterns = [
            r"\d{4}\s\d{4}\s\d{4}",
            r"\d{12}",
            r"\d{4}-\d{4}-\d{4}"
        ]

        uid_found = any(re.search(p, text) for p in uid_patterns)

        keywords = [
            "aadhaar",
            "uidai",
            "government of india",
            "year of birth",
            "male",
            "female"
        ]

        keyword_score = sum(word in text.lower() for word in keywords)

        # RELAXED CONDITION
        return uid_found or keyword_score >= 2

    except Exception as e:
        st.error(f"OCR Error: {str(e)}")
        return False

# =========================
# FACE DETECTION (SAFE)
# =========================
def detect_face(image):
    try:
        if face_cascade.empty():
            return False

        img = np.array(image)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        return len(faces) > 0

    except:
        return False

# =========================
# LOGO DETECTION
# =========================
def detect_logo(image):
    img = np.array(image)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    lower_orange = np.array([5, 100, 100])
    upper_orange = np.array([20, 255, 255])

    mask = cv2.inRange(hsv, lower_orange, upper_orange)

    return np.sum(mask) > 500

# =========================
# FRAUD DETECTION (LIGHTWEIGHT)
# =========================
def detect_fraud(image):
    img = np.array(image)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0

    gray = cv2.cvtColor((img*255).astype("uint8"), cv2.COLOR_RGB2GRAY)
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
# REPORT GENERATION
# =========================
def generate_report(record):
    df = pd.DataFrame([record])
    return df.to_csv(index=False).encode("utf-8")

# =========================
# PAGE: DETECT
# =========================
if page == "Detect":

    st.title("🪪 Aadhaar Fraud Detection System")

    option = st.radio("Select Input Method:", ["Upload Image", "Use Camera"])

    image = None

    if option == "Upload Image":
        file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])
        if file:
            image = Image.open(file)

    else:
        file = st.camera_input("Capture Image")
        if file:
            image = Image.open(file)

    if image:
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Checks
        aadhaar_flag = is_aadhaar(image)
        face_flag = detect_face(image)
        logo_flag = detect_logo(image)

        st.write("🔍 Aadhaar Text Check:", aadhaar_flag)
        st.write("🙂 Face Detected:", face_flag)
        st.write("🎨 Logo Detected:", logo_flag)

        # FINAL DECISION (IMPROVED LOGIC)
        if not (aadhaar_flag or logo_flag):
            result = "NOT AADHAAR ❌"
            st.error(result)

        elif not face_flag:
            result = "NO FACE FOUND ❌"
            st.warning(result)

        else:
            result = detect_fraud(image)

            if result == "FRAUD":
                st.error("⚠️ Fraudulent Aadhaar Detected")
            elif result == "GENUINE":
                st.success("✅ Genuine Aadhaar")
            else:
                st.warning("❌ Invalid Image")

        st.write("Final Result:", result)

        # Save to history
        record = {
            "Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Result": result
        }

        st.session_state.history.append(record)

        # Download report
        st.download_button(
            "Download Report",
            data=generate_report(record),
            file_name="aadhaar_report.csv",
            mime="text/csv"
        )

# =========================
# PAGE: HISTORY
# =========================
elif page == "History":

    st.title("📜 Detection History")

    if st.session_state.history:
        df = pd.DataFrame(st.session_state.history)
        st.dataframe(df)
    else:
        st.write("No history available.")

    if st.button("Clear History"):
        st.session_state.history = []
        st.success("History Cleared!")
