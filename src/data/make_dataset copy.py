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
def preprocess_image(filepath, label):
    # Read and decode image
    image = tf.io.read_file(filepath)
    try:
        image = tf.image.decode_jpeg(image, channels=3)  # Decode as JPEG (assumes your images are JPEG)
    except tf.errors.InvalidArgumentError:
        print(f"Error decoding image: {filepath}")
        return None, label
    image = tf.image.resize(image, IMAGE_SIZE)  # Resize image
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)  # Preprocess for MobileNetV2
    return image, label

# --------------------------------------------------------------
# Create Dataset
# --------------------------------------------------------------
def create_dataset(dataframe, x_col, y_col, batch_size, shuffle=False):
    # Validate file paths
    dataframe = dataframe[dataframe[x_col].apply(os.path.exists)]
    
    # Create TensorFlow dataset
    filepaths = dataframe[x_col].values
    labels = dataframe[y_col].values
    dataset = tf.data.Dataset.from_tensor_slices((filepaths, labels))
    
    # Apply preprocessing
    dataset = dataset.map(preprocess_image, num_parallel_calls=AUTOTUNE)
    
    # Remove None values (from failed preprocessing)
    dataset = dataset.filter(lambda img, label: img is not None)
    
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
    print("Batch labels:", labels)

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