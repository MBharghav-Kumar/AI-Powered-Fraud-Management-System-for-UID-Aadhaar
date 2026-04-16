import streamlit as st
import numpy as np
import cv2
import pandas as pd
from tensorflow.keras.models import load_model
from PIL import Image

# Load model
model = load_model(
    r"D:\NMIMS\4th year\internship\Springboard\fraud_model.h5")

# Load OCR CSV
ocr_df = pd.read_csv(r"D:\NMIMS\4th year\internship\Springboard\ocr_results.csv")

IMG_SIZE = 128

# Title
st.title("AI Powered Aadhaar Fraud Detection System")
st.write("Upload an Aadhaar image to check if it is Genuine or Fraud.")

# Upload file
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

def cnn_predict(image):
    img = np.array(image)

    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2EGB)

    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    img = np.reshape(img, (1, IMG_SIZE, IMG_SIZE, 3))

    pred = model.predict(img)[0][0]

    return pred

#    return "FRAUD" if pred > 0.5 else "GENUINE"

# OCR Validation (from CSV)

def ocr_validate(file_name):
    row = ocr_df[ocr_df["file"] == file_name]

    if row.empty:
        return False, "No OCR data found"
    
    uid = row.iloc[0]["uid"]
    dob = row.iloc[0]["dob"]

    # Basic validation
    if str(uid) == "[]" or str(dob) == "[]":
        return False, "Missing UID or DOB"
    
    return True, "Valid OCR Data"

# When user uploads
if uploaded_file is not None:
    try:

        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Save uploaded file temporarily
        file_name = uploaded_file.name

        #CNN result
        cnn_score = cnn_predict(image)

        #OCR validation
        ocr_valid, ocr_msg = ocr_validate(file_name)

        # Debug info
        st.write("CNN Score:", cnn_score)
        st.write("OCR Status:", ocr_msg)

        # FINAL DECISION
        if cnn_score > 0.5 or not ocr_valid:
            st.error("Fraudulent Document Detected!")
            result = "FRAUD"
        else:
            st.success("Genuine Document")
            result = "GENUINE"
        
        st.write("Final Prediction:", result)

    except Exception as e:
        st.error(f"Error: {str(e)}")
