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
# FACE DETECTION
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
# QR DETECTION
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
# IMPROVED OCR (UID CHECK)
# =========================
def is_aadhaar(image):
    try:
        gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)

        # Improve OCR
        gray = cv2.GaussianBlur(gray, (3,3), 0)
        gray = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)[1]

        text = pytesseract.image_to_string(gray)
        st.write("🔍 OCR Text:", text)

        uid_pattern = r"\d{4}\s?\d{4}\s?\d{4}"
        uid_match = re.findall(uid_pattern, text)

        keywords = ["aadhaar", "uidai", "government of india"]
        keyword_found = any(word in text.lower() for word in keywords)

        return bool(uid_match and keyword_found)

    except:
        st.warning("⚠️ OCR failed")
        return False

# =========================
# LIGHT VALIDATION (NO FAKE MISCLASSIFICATION)
# =========================
def detect_quality(image):
    img = np.array(image)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.sum(edges) / (128*128)

    if edge_density < 1:
        return "LOW_QUALITY"
    else:
        return "OK"

# =========================
# REPORT
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
    table.setStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.grey),
        ("TEXTCOLOR",(0,0),(-1,0),colors.white),
        ("GRID", (0,0), (-1,-1), 1, colors.black)
    ])

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

        img_face, face_flag = detect_face(img_np)
        img_qr, qr_flag = detect_qr(img_face)

        st.image(img_qr, caption="Detected Features")

        aadhaar_flag = is_aadhaar(image)
        logo_flag = detect_logo(image)
        quality_flag = detect_quality(image)

        st.write("Face:", face_flag)
        st.write("QR:", qr_flag)
        st.write("Logo:", logo_flag)
        st.write("UID Detected:", aadhaar_flag)
        st.write("Quality:", quality_flag)

        # =========================
        # FINAL DECISION
        # =========================
        if not aadhaar_flag:
            result = "NOT AADHAAR ❌"
            st.error("❌ UID not detected")

        elif not face_flag:
            result = "FAKE AADHAAR ❌"
            st.error("❌ No face detected")

        elif quality_flag == "LOW_QUALITY":
            result = "SUSPICIOUS ⚠️"
            st.warning("⚠️ Low quality image")

        else:
            result = "GENUINE ✅"
            st.success("✅ Valid Aadhaar")

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
