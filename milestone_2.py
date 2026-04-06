#import pytesseract
#pytesseract.pytesseract.tesseract_cmd = r"D:\NMIMS\4th year\internship\Springboard\tesseract-main\tesseract.exe"

import cv2
import pytesseract
import re
import pandas as pd
import os

# Set Tesseract path
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

folder_path = r"D:\NMIMS\4th year\internship\Springboard\archive"

results = []

def extract_fields(text):
    # Aadhaar Number (12 digits)
    uid = re.findall(r"\b\d{4}\s\d{4}\s\d{4}\b", text)

    # DOB (dd/mm/yyyy)
    dob = re.findall(r"\b\d{2}/\d{2}/\d{4}\b", text)

    # Gender
    gender = "Male" if "Male" in text else ("Female" if "Female" in text else "")

    # Name (simple assumption: first line)
    lines = text.split("\n")
    name = lines[0] if lines else ""

    return name, uid, dob, gender

def validate(uid, dob):
    uid_valid = len(uid) > 0
    dob_valid = len(dob) > 0

    return uid_valid and dob_valid

print("Starting OCR Processing...\n")

count = 0

for file in os.listdir(folder_path):
    if file.endswith(".jpg"):

        img_path = os.path.join(folder_path, file)
        img = cv2.imread(img_path)

        if img is None:
            continue

        # OCR
        text = pytesseract.image_to_string(img)

        name, uid, dob, gender = extract_fields(text)

        is_valid = validate(uid, dob)

        results.append({
            "file": file,
            "name": name,
            "uid": uid,
            "dob": dob,
            "gender": gender,
            "valid": is_valid
        })

        count += 1
        if count > 200:  # limit for testing
            break

df = pd.DataFrame(results)
df.to_csv("ocr_results.csv", index=False)

print("OCR Completed!")
print("Results saved to ocr_results.csv")