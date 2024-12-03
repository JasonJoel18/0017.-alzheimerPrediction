import pandas as pd
from glob import glob
from pathlib import Path
import os
import functions
import cv2
import numpy as np
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# --------------------------------------------------------------
# Functions
# --------------------------------------------------------------

def contour_and_crop(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    return image[y:y+h, x:x+w]

# Resize and normalize image
def preprocess_image(image, target_size=(128, 128)):
    cropped_image = contour_and_crop(image)
    resized_image = cv2.resize(cropped_image, target_size, interpolation=cv2.INTER_AREA)
    normalized_image = resized_image / 255.0
    return normalized_image

# Load dataset and preprocess
def load_data(data_dir, target_size=(128, 128)):
    images = []
    labels = []
    classes = os.listdir(data_dir)
    for label, class_name in enumerate(classes):
        if class_name == '.gitkeep':
            continue
        class_dir = os.path.join(data_dir, class_name)
        for file in os.listdir(class_dir):
            img_path = os.path.join(class_dir, file)
            img = cv2.imread(img_path)
            if img is not None:
                processed_img = preprocess_image(img, target_size)
                images.append(processed_img)
                labels.append(class_name)  # Store the class name directly
    return np.array(images), np.array(labels)

# --------------------------------------------------------------
# Preprocess data
# --------------------------------------------------------------
data_dir = "/Volumes/Jason's T7/2. Education/Research/Thesis/Paper/0017. alzheimerPrediction/data2/external"
X, y = load_data(data_dir)

# Convert labels to numeric values
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)
# y_categorical = to_categorical(y_encoded)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# One-hot encode labels
# y_train = to_categorical(y_train)
# y_val = to_categorical(y_val)
# y_test = to_categorical(y_test)

print(f"Training data shape: {X_train.shape}")
print(f"Validation data shape: {X_val.shape}")
print(f"Test data shape: {X_test.shape}")

# --------------------------------------------------------------
# View Images
# --------------------------------------------------------------

# Function to display images with labels
def display_images(images, labels, class_names, num_images=10):
    # Convert one-hot encoded labels back to integers
    if len(labels.shape) > 1 and labels.shape[1] > 1:  # Check if labels are one-hot encoded
        labels = np.argmax(labels, axis=1)
    
    fig, axes = plt.subplots(nrows=2, ncols=6, figsize=(15, 6))
    for i in range(num_images):
        ax = axes.flat[i]
        ax.imshow(images[i])
        ax.set_title(f"{class_names[labels[i]]}")  # Display label name
        ax.axis('off')
    plt.tight_layout()
    plt.show()

# Example usage:
display_images(X_train, y_train, class_names=encoder.classes_, num_images=12)

# --------------------------------------------------------------
# Export dataset
# --------------------------------------------------------------



# --------------------------------------------------------------
# Export dataset
# --------------------------------------------------------------
