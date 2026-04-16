import os
import cv2
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split

# Paths
image_folder = r"D:\NMIMS\4th year\internship\Springboard\archive\processed_images"
csv_path = os.path.join(image_folder, "metadata.csv")

IMG_SIZE = 64

# Load data
df = pd.read_csv(csv_path)

images = []
labels = []

print("Loading images...")

for _, row in df.iterrows():
    img_path = os.path.join(image_folder, row['file_name'])

    img = cv2.imread(img_path)
    if img is None:
        continue

    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    images.append(img)
    labels.append(1 if row['label'] == 'fraud' else 0)

X = np.array(images) / 255.0
y = np.array(labels)

# Reshape for CNN
X = X.reshape(-1, 1, IMG_SIZE, IMG_SIZE)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Convert to tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)

X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

# CNN Model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 14 * 14, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x

model = CNN()

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training
print("Training started...")

for epoch in range(5):
    outputs = model(X_train).squeeze()
    loss = criterion(outputs, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch+1}, Loss: {loss.item()}")

print("Training complete!")

# Save model
torch.save(model.state_dict(), "fraud_model.pth")
