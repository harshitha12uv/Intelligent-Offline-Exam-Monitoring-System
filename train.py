import cv2
import os
import numpy as np

def load_dataset():
    images, labels = [], []
    dataset_path = 'dataset/'
    for idx, category in enumerate(['Normal', 'Suspicious']):
        category_path = os.path.join(dataset_path, category)
        for filename in os.listdir(category_path):
            img = cv2.imread(os.path.join(category_path, filename), cv2.IMREAD_GRAYSCALE)
            images.append(img)
            labels.append(idx)
    return images, np.array(labels)

def train_model():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    images, labels = load_dataset()
    recognizer.train(images, labels)
    recognizer.save('face_recognizer.yml')
    print("Model trained and saved!")

if __name__ == "__main__":
    train_model()
