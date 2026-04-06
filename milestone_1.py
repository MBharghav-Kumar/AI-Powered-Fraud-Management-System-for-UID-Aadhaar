import os
import cv2
import numpy as np
import pandas as pd
import random

input_folder = r"D:\NMIMS\4th year\internship\Springboard\archive"
output_folder = r"D:\NMIMS\4th year\internship\Springboard\archive\processed_images"

os.makedirs(output_folder, exist_ok=True)

TARGET_SIZE = (224, 224)
AUGMENTATIONS_PER_IMAGE = 5

metadata = []

def augment_image(image):
    rows, cols = image.shape

    # Rotation
    angle = random.uniform(-5, 5)
    M = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
    rotated = cv2.warpAffine(image, M, (cols, rows))

    # Brightness
    brightness = random.uniform(0.9, 1.1)
    bright = cv2.convertScaleAbs(rotated, alpha=brightness, beta=0)

    # Slight shift
    tx = random.randint(-5,5)
    ty = random.randint(-5, 5)
    M_shift = np.float32([[1,0, tx], [0,1,ty]])
    shifted = cv2.warpAffine(bright, M_shift, (cols, rows))

    return shifted

print("Processing Started...")

for file in os.listdir(input_folder):
    if file.lower().endswith((".jpg", ".jpeg", ".png")):

        img_path = os.path.join(input_folder, file)
        img = cv2.imread(img_path)

        if img is None:
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, TARGET_SIZE)
   #     blurred = cv2.GaussianBlur(resized, (3, 3), 0)

        processed = resized

        base_name = os.path.splitext(file)[0]

        # Save original
        orig_name = base_name + "_orig.jpg"
        cv2.imwrite(os.path.join(output_folder, orig_name), processed)

        metadata.append({"file_name": orig_name, "label": "genuine"})

        # Augment
        for i in range(AUGMENTATIONS_PER_IMAGE):
            aug_img = augment_image(processed)

            aug_name = f"{base_name}_aug_{i}.jpg"
            cv2.imwrite(os.path.join(output_folder, aug_name), aug_img)

      #      label = "fraud" if random.random() < 0.2 else "genuine"
            metadata.append({"file_name": aug_name, "label": "genuine"})

metadata_df = pd.DataFrame(metadata)
metadata_df.to_csv(os.path.join(output_folder, "metadata.csv"), index=False)

print("Completed Successfully!")
print("Total Images Generated:", len(metadata_df))