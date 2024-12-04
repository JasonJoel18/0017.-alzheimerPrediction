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
import tensorflow as tf

# --------------------------------------------------------------
# Functions
# --------------------------------------------------------------

# --------------------------------------------------------------
# Preprocess data
# --------------------------------------------------------------
dir = {
    'Mild Demented': '/Volumes/Jason\'s T7/2. Education/Research/Thesis/Paper/0017. alzheimerPrediction/data2/external/MildDemented',
    'Moderate Demented': '/Volumes/Jason\'s T7/2. Education/Research/Thesis/Paper/0017. alzheimerPrediction/data2/external/ModerateDemented',
    'Non Demented': '/Volumes/Jason\'s T7/2. Education/Research/Thesis/Paper/0017. alzheimerPrediction/data2/external/NonDemented',
    'Very MildDemented': '/Volumes/Jason\'s T7/2. Education/Research/Thesis/Paper/0017. alzheimerPrediction/data2/external/VeryMildDemented'
}
img_path = []
lbl = []
for cls, dir in dir.items():
    for img in os.listdir(dir):
        img_path.append(os.path.join(dir, img))
        lbl.append(cls)

print(img_path)
print(lbl)

dt = pd.DataFrame({'img_path': img_path, 'lbl': lbl})
print(dt.head())
print(dt['lbl'].value_counts())

# Train-test split
train_path, test_path = train_test_split(dt, test_size=0.4, random_state=42)
test_path, val_path = train_test_split(test_path, test_size=0.5, random_state=42)


print(f"Training data shape: {train_path.shape}")
print(f"Validation data shape: {val_path.shape}")
print(f"Test data shape: {test_path.shape}")

# --------------------------------------------------------------
# View Images
# --------------------------------------------------------------


label_mapping = {'Non Demented': 0, 'Moderate Demented': 1, 'Mild Demented': 2, 'Very MildDemented': 3}
print("Label Mapping:", label_mapping)

train_path['lbl'] = train_path['lbl'].map(label_mapping)
val_path['lbl'] = val_path['lbl'].map(label_mapping)
test_path['lbl'] = test_path['lbl'].map(label_mapping)

img_size = (224, 224)
btc_size = 32


# Define constants
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
AUTOTUNE = tf.data.AUTOTUNE

# Preprocessing function
import cv2
import numpy as np



# --------------------------------------------------------------
# Create Dataset
# --------------------------------------------------------------


def preprocess_image_with_opencv(filepath, label):
    def preprocess(filepath):
        # Read the image using OpenCV
        filepath = filepath.decode('utf-8')  # Decode the TensorFlow string tensor to Python string
        image = cv2.imread(filepath)
        if image is None:
            raise ValueError(f"Image at path {filepath} could not be read.")
        
        # Convert to grayscale for contour detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            # Find the largest contour and get its bounding box
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            image = image[y:y+h, x:x+w]  # Crop the image to the bounding box
        
        # Resize the image to the target size
        image = cv2.resize(image, IMAGE_SIZE)
        
        # Normalize the image for MobileNetV2
        image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
        return image
    
    # Use tf.numpy_function to apply the NumPy-based preprocessing
    image = tf.numpy_function(preprocess, [filepath], tf.float32)
    image.set_shape(IMAGE_SIZE + (3,))
    label = tf.one_hot(label, depth=4)  # One-hot encode labels
    return image, label

def create_dataset(dataframe, x_col, y_col, batch_size, shuffle=False):
    # Validate file paths
    dataframe = dataframe[dataframe[x_col].apply(os.path.exists)]
    
    # Create TensorFlow dataset
    filepaths = dataframe[x_col].values
    labels = dataframe[y_col].values
    dataset = tf.data.Dataset.from_tensor_slices((filepaths, labels))
    
    # Apply preprocessing
    dataset = dataset.map(preprocess_image_with_opencv, num_parallel_calls=AUTOTUNE)
    
    # Shuffle, batch, and prefetch
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(dataframe))
    dataset = dataset.batch(batch_size).prefetch(AUTOTUNE)
    return dataset


# Create datasets
train_dataset = create_dataset(train_path, x_col="img_path", y_col="lbl", batch_size=BATCH_SIZE, shuffle=True)
test_dataset = create_dataset(test_path, x_col="img_path", y_col="lbl", batch_size=BATCH_SIZE, shuffle=False)
val_dataset = create_dataset(val_path, x_col="img_path", y_col="lbl", batch_size=BATCH_SIZE, shuffle=False)
# Print label mapping
print("Label Mapping:", label_mapping)

# Debugging
for images, labels in train_dataset.take(1):
    print("Batch images shape:", images.shape)
    print("Batch labels:", labels.shape)

print("Datasets created successfully!")

# ================================================================










# =================================================================


def show_images(dataset, label_mapping, max_images=25):
    """
    Display a batch of images with their corresponding labels.
    
    Args:
    - dataset (tf.data.Dataset): TensorFlow dataset to visualize.
    - label_mapping (dict): Dictionary mapping numeric labels to class names.
    - max_images (int): Maximum number of images to display.
    """
    # Reverse the label mapping for lookup
    reverse_label_mapping = {v: k for k, v in label_mapping.items()}
    
    # Get a batch of images and labels
    for images, labels in dataset.take(1):  # Take one batch from the dataset
        plt.figure(figsize=(20, 20))
        num_images = min(max_images, len(labels))
        for i in range(num_images):
            plt.subplot(5, 5, i + 1)
            image = (images[i].numpy() + 1) / 2  # Rescale images to [0, 1] if necessary
            plt.imshow(image)
            class_name = reverse_label_mapping[labels[i].numpy()]
            plt.title(class_name, color="green", fontsize=16)
            plt.axis('off')
        plt.show()
        break
    
# Visualize training dataset
show_images(train_dataset, label_mapping)