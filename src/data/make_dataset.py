import pandas as pd
from glob import glob
from pathlib import Path
import os
import functions
import cv2
from sklearn.model_selection import train_test_split
import tensorflow as tf
print("TensorFlow version:", tf.__version__)


# --------------------------------------------------------------
# Constants and Configuration
# --------------------------------------------------------------
IMAGE_SIZE = (224,224)
BATCH_SIZE = 32
AUTOTUNE = tf.data.AUTOTUNE


# Class directories
dir = {
    'Mild Demented': '/Volumes/Jason\'s T7/2. Education/Research/Thesis/Paper/0017. alzheimerPrediction/data2/external/MildDemented',
    'Moderate Demented': '/Volumes/Jason\'s T7/2. Education/Research/Thesis/Paper/0017. alzheimerPrediction/data2/external/ModerateDemented',
    'Non Demented': '/Volumes/Jason\'s T7/2. Education/Research/Thesis/Paper/0017. alzheimerPrediction/data2/external/NonDemented',
    'Very MildDemented': '/Volumes/Jason\'s T7/2. Education/Research/Thesis/Paper/0017. alzheimerPrediction/data2/external/VeryMildDemented'
}

label_mapping = {'Non Demented': 0, 'Moderate Demented': 1,
                 'Mild Demented': 2, 'Very MildDemented': 3}
print("Label Mapping:", label_mapping)


# --------------------------------------------------------------
# Data Preparation
# --------------------------------------------------------------
# Collect image paths and labels
img_path = []
lbl = []
for cls, dir in dir.items():
    for img in os.listdir(dir):
        img_path.append(os.path.join(dir, img))
        lbl.append(cls)

# print(img_path)
# print(lbl)

dt = pd.DataFrame({'img_path': img_path, 'lbl': lbl})
# print(dt.head())
# print(dt['lbl'].value_counts())
dt['lbl'] = dt['lbl'].map(label_mapping)


# Train-test split
train_path, test_path = train_test_split(dt, test_size=0.4, random_state=42)
test_path, val_path = train_test_split(test_path, test_size=0.5, random_state=42)


print(f"Training data shape: {train_path.shape}")
print(f"Validation data shape: {val_path.shape}")
print(f"Test data shape: {test_path.shape}")


# --------------------------------------------------------------
# Create Dataset
# --------------------------------------------------------------


def preprocess_image_with_opencv(filepath, label):
    def preprocess(filepath):
        filepath = filepath.decode('utf-8')
        image = cv2.imread(filepath)
        if image is None:
            raise ValueError(f"Image at path {filepath} could not be read.")
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            image = image[y:y+h, x:x+w]
        image = cv2.resize(image, IMAGE_SIZE)
        image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
        return image
    image = tf.numpy_function(preprocess, [filepath], tf.float32)
    image.set_shape(IMAGE_SIZE + (3,))
    label = tf.one_hot(label, depth=4)
    return image, label


def create_dataset(dataframe, x_col, y_col, batch_size, shuffle=False):
    dataframe = dataframe[dataframe[x_col].apply(os.path.exists)]
    filepaths = dataframe[x_col].values
    labels = dataframe[y_col].values
    dataset = tf.data.Dataset.from_tensor_slices((filepaths, labels))
    dataset = dataset.map(preprocess_image_with_opencv,
                          num_parallel_calls=AUTOTUNE)
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(dataframe))
    dataset = dataset.batch(batch_size).prefetch(AUTOTUNE)
    return dataset

train_dataset = create_dataset(train_path, x_col="img_path", y_col="lbl", batch_size=BATCH_SIZE, shuffle=True)
test_dataset = create_dataset(test_path, x_col="img_path", y_col="lbl", batch_size=BATCH_SIZE, shuffle=False)
val_dataset = create_dataset(val_path, x_col="img_path", y_col="lbl", batch_size=BATCH_SIZE, shuffle=False)


# Debugging
for images, labels in train_dataset.take(1):
    print("Batch images shape:", images.shape)
    print("Batch labels:", labels.shape)

import matplotlib.pyplot as plt
import numpy as np
import cv2

def display_images(images, labels, class_names, num_images=10):
    """
    Function to display a batch of images along with their labels.

    Arguments:
    - images: Batch of images (as a TensorFlow tensor or numpy array).
    - labels: Corresponding labels (either as integers or one-hot encoded).
    - class_names: List of class names corresponding to each label.
    - num_images: Number of images to display from the batch.
    """
    # Convert one-hot encoded labels back to integers if needed
    if len(labels.shape) > 1 and labels.shape[1] > 1:  # Check if labels are one-hot encoded
        labels = np.argmax(labels, axis=1)
    
    # Rescale images to be in the [0, 1] range (from [-1, 1] or other ranges)
    images = (images + 1) / 2  # Rescale from [-1, 1] to [0, 1]
    
    # Set up the plot grid
    rows = (num_images // 6) + (1 if num_images % 6 != 0 else 0)
    fig, axes = plt.subplots(nrows=rows, ncols=6, figsize=(15, 6))
    axes = axes.flatten()  # Flatten the axes to make it easy to index
    
    # Loop over the images and display them
    for i in range(num_images):
        ax = axes[i]
        img = images[i].numpy().astype("float32")  # Convert TensorFlow tensor to numpy array if needed
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB for display
        ax.imshow(img)
        ax.set_title(f"{class_names[labels[i]]}")  # Display the class name
        ax.axis('off')
    
    # Remove unused axes if the number of images is less than the number of subplots
    for i in range(num_images, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()

class_names = ['Non Demented', 'Moderate Demented', 'Mild Demented', 'Very Mild Demented']
# Fetch a batch from the dataset
for images, labels in train_dataset.take(1):
    # Display the first 12 images from the batch
    display_images(images, labels, class_names, num_images=12)
    
print("Datasets created successfully!")