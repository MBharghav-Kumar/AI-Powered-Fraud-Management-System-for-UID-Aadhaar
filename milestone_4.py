import streamlit as st
import numpy as np
import cv2
from PIL import Image
import pandas as pd
from datetime import datetime
import pytesseract
import os
import re
from io import BytesIO

# =========================
# TESSERACT CONFIG (LOCAL ONLY)
# =========================
try:
    if os.name == "nt":
        pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
except:
    pass

IMG_SIZE = 128

# Load face cascade
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="Aadhaar Fraud Detection", layout="centered")

# =========================
# SESSION HISTORY
# =========================
if "history" not in st.session_state:
    st.session_state.history = []

# =========================
# SIDEBAR
# =========================
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Detect", "History"])

# =========================
# FACE DETECTION (BOX)
# =========================
def detect_face(image):
    img = np.array(image)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0,255,0), 2)
        cv2.putText(img, "FACE", (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

    return img, len(faces) > 0

# =========================
# QR DETECTION (IMPROVED)
# =========================
def detect_qr(image):
    img = np.array(image)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray = cv2.equalizeHist(gray)

    detector = cv2.QRCodeDetector()

    data, bbox, _ = detector.detectAndDecode(gray)

    if bbox is not None:
        bbox = bbox.astype(int)

        for i in range(len(bbox[0])):
            pt1 = tuple(bbox[0][i])
            pt2 = tuple(bbox[0][(i+1) % len(bbox[0])])
            cv2.line(img, pt1, pt2, (0,255,0), 2)

        cv2.putText(img, "QR", tuple(bbox[0][0]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        return img, True

    return img, False

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
# STRICT AADHAAR CHECK (UID REQUIRED)
# =========================
def is_aadhaar(image):
    try:
        text = pytesseract.image_to_string(image)
        st.write("🔍 OCR Text:", text)

        # STRICT UID patterns
        uid_patterns = [
            r"\b\d{4}\s\d{4}\s\d{4}\b",
            r"\b\d{12}\b",
            r"\b\d{4}-\d{4}-\d{4}\b"
        ]

        uid_found = any(re.search(p, text) for p in uid_patterns)

        keywords = [
            "aadhaar",
            "uidai",
            "government of india"
        ]

        keyword_found = any(word in text.lower() for word in keywords)

        return uid_found and keyword_found

    except:
        st.warning("⚠️ OCR not available")
        return False

# =========================
# FRAUD DETECTION
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
# REPORT FUNCTIONS
# =========================
def generate_csv(df):
    return df.to_csv(index=False).encode("utf-8")

def generate_pdf(df):
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle
    from reportlab.lib import colors

    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer)

    data = [df.columns.tolist()] + df.values.tolist()

    table = Table(data)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.grey),
        ("TEXTCOLOR",(0,0),(-1,0),colors.white),
        ("GRID", (0,0), (-1,-1), 1, colors.black)
    ]))

    doc.build([table])
    buffer.seek(0)
    return buffer

# =========================
# DETECT PAGE
# =========================
if page == "Detect":

    st.title("🪪 Aadhaar Fraud Detection System")

    option = st.radio("Select Input Method:", ["Upload Image", "Use Camera"])

    image = None

    if option == "Upload Image":
        file = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])
        if file:
            image = Image.open(file)
    else:
        file = st.camera_input("Capture Image")
        if file:
            image = Image.open(file)

    if image:
        st.image(image, caption="Original Image")

        user_name = st.text_input("Enter Name (as per Aadhaar):")

        if not user_name:
            st.warning("Please enter name")
            st.stop()

        img_np = np.array(image)

        # FACE + QR VISUAL
        img_face, face_flag = detect_face(img_np)
        img_qr, qr_flag = detect_qr(img_face)

        st.image(img_qr, caption="Detected Features")

        # CHECKS
        aadhaar_flag = is_aadhaar(image)
        logo_flag = detect_logo(image)

        st.write("Face:", face_flag)
        st.write("QR:", qr_flag)
        st.write("Logo:", logo_flag)
        st.write("UID Valid:", aadhaar_flag)

        # =========================
        # FINAL DECISION (STRICT)
        # =========================
        if not aadhaar_flag:
            result = "NOT AADHAAR ❌"
            st.error("❌ UID not detected — Not Aadhaar")

        elif not face_flag:
            result = "FAKE AADHAAR ❌"
            st.error("❌ No face detected — Invalid Aadhaar")

        else:
            result = detect_fraud(image)

            if result == "FRAUD":
                st.error("⚠️ Fraudulent Aadhaar Detected")
            elif result == "GENUINE":
                st.success("✅ Genuine Aadhaar")
            else:
                st.warning("❌ Invalid Image")

        st.write("Final Result:", result)

        # SAVE HISTORY
        record = {
            "Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Name": user_name,
            "Result": result
        }

        st.session_state.history.append(record)

# =========================
# HISTORY PAGE
# =========================
elif page == "History":

    st.title("📜 Detection History")

    if st.session_state.history:
        df = pd.DataFrame(st.session_state.history)
        st.dataframe(df)

        st.download_button("⬇️ Download CSV", generate_csv(df), "history.csv")

        st.download_button("⬇️ Download PDF", generate_pdf(df), "history.pdf")

    else:
        st.write("No history available.")

    if st.button("Clear History"):
        st.session_state.history = []
        st.success("History cleared!")
