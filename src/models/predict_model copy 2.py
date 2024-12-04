import pandas as pd
import os
import cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten

# --------------------------------------------------------------
# Constants and Configuration
# --------------------------------------------------------------
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
AUTOTUNE = tf.data.AUTOTUNE

# Class directories
dir_paths = {
    'Mild Demented': '/Volumes/Jason\'s T7/2. Education/Research/Thesis/Paper/0017. alzheimerPrediction/data2/external/MildDemented',
    'Moderate Demented': '/Volumes/Jason\'s T7/2. Education/Research/Thesis/Paper/0017. alzheimerPrediction/data2/external/ModerateDemented',
    'Non Demented': '/Volumes/Jason\'s T7/2. Education/Research/Thesis/Paper/0017. alzheimerPrediction/data2/external/NonDemented',
    'Very MildDemented': '/Volumes/Jason\'s T7/2. Education/Research/Thesis/Paper/0017. alzheimerPrediction/data2/external/VeryMildDemented'
}

label_mapping = {
    'Non Demented': 0,
    'Moderate Demented': 1,
    'Mild Demented': 2,
    'Very Mild Demented': 3
}

# --------------------------------------------------------------
# Data Preparation
# --------------------------------------------------------------
# Collect image paths and labels
img_paths, labels = [], []
for cls, dir_path in dir_paths.items():
    for img in os.listdir(dir_path):
        img_paths.append(os.path.join(dir_path, img))
        labels.append(cls)

data = pd.DataFrame({'img_path': img_paths, 'lbl': labels})
data['lbl'] = data['lbl'].map(label_mapping)

# Train-test split
train_data, test_data = train_test_split(data, test_size=0.4, random_state=42)
test_data, val_data = train_test_split(test_data, test_size=0.5, random_state=42)

print(f"Training data shape: {train_data.shape}")
print(f"Validation data shape: {val_data.shape}")
print(f"Test data shape: {test_data.shape}")

# --------------------------------------------------------------
# Dataset Creation
# --------------------------------------------------------------
def preprocess_image_with_opencv(filepath, label):
    def preprocess(filepath):
        filepath = filepath.decode('utf-8')
        image = cv2.imread(filepath)
        if image is None:
            raise ValueError(f"Image at path {filepath} could not be read.")
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
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
    dataset = dataset.map(preprocess_image_with_opencv, num_parallel_calls=AUTOTUNE)
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(dataframe))
    dataset = dataset.batch(batch_size).prefetch(AUTOTUNE)
    return dataset

# Create datasets
train_dataset = create_dataset(train_data, x_col="img_path", y_col="lbl", batch_size=BATCH_SIZE, shuffle=True)
val_dataset = create_dataset(val_data, x_col="img_path", y_col="lbl", batch_size=BATCH_SIZE, shuffle=False)
test_dataset = create_dataset(test_data, x_col="img_path", y_col="lbl", batch_size=BATCH_SIZE, shuffle=False)

# --------------------------------------------------------------
# Model Definition
# --------------------------------------------------------------
base_model = tf.keras.applications.Xception(include_top=False, weights="imagenet", input_shape=IMAGE_SIZE + (3,), pooling='max')

model = Sequential([
    base_model,
    Flatten(),
    Dropout(rate=0.3),
    Dense(128, activation='relu'),
    Dropout(rate=0.25),
    Dense(4, activation='softmax')  # Output layer for 4 classes
])

model.compile(
    optimizer=tf.keras.optimizers.Adamax(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# --------------------------------------------------------------
# Model Training
# --------------------------------------------------------------
history = model.fit(
    train_dataset,
    epochs=10,
    validation_data=val_dataset,
    validation_freq=1
)

# --------------------------------------------------------------
# Model Evaluation
# --------------------------------------------------------------
test_loss, test_accuracy = model.evaluate(test_dataset)
print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy}")