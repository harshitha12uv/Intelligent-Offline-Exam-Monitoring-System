import cv2
import numpy as np
import os

# Path to dataset
dataset_path = 'dataset/'
label_names = ['Normal', 'Suspicious']
model_path = 'face_recognizer.yml'

# Function to load images and labels
def load_dataset():
    images = []
    labels = []

    for label, category in enumerate(label_names):
        category_path = os.path.join(dataset_path, category)
        for filename in os.listdir(category_path):
            img_path = os.path.join(category_path, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                images.append(img)
                labels.append(label)

    return images, np.array(labels)

# Train and save the model
def train_model():
    print("[INFO] Loading dataset...")
    images, labels = load_dataset()
    print(f"[INFO] Dataset loaded: {len(images)} images")

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    print("[INFO] Training the model...")
    recognizer.train(images, labels)
    recognizer.save(model_path)
    print(f"[INFO] Model saved to {model_path}")

if __name__ == "__main__":
    train_model()
