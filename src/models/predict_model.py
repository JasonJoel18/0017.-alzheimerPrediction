import pandas as pd
from glob import glob
from pathlib import Path
import os
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

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

dt = pd.DataFrame({'img_path': img_path, 'lbl': lbl})
print(dt['lbl'].value_counts())

# Train-test split
train_path, test_path = train_test_split(dt, test_size=0.4, random_state=42)
test_path, val_path = train_test_split(test_path, test_size=0.5, random_state=42)

print(f"Training data shape: {train_path.shape}")
print(f"Validation data shape: {val_path.shape}")
print(f"Test data shape: {test_path.shape}")

# Map class labels to integers
label_mapping = {'Non Demented': 0, 'Moderate Demented': 1, 'Mild Demented': 2, 'Very MildDemented': 3}
train_path['lbl'] = train_path['lbl'].map(label_mapping)
val_path['lbl'] = val_path['lbl'].map(label_mapping)
test_path['lbl'] = test_path['lbl'].map(label_mapping)

# --------------------------------------------------------------
# Create Dataset
# --------------------------------------------------------------
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
AUTOTUNE = tf.data.AUTOTUNE

# Preprocessing function
def preprocess_image(filepath, label):
    image = tf.io.read_file(filepath)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, IMAGE_SIZE)
    image = tf.keras.applications.xception.preprocess_input(image)
    label = tf.one_hot(label, depth=4)  # One-hot encode labels
    return image, label

def create_dataset(dataframe, x_col, y_col, batch_size, shuffle=False):
    dataframe = dataframe[dataframe[x_col].apply(os.path.exists)]
    filepaths = dataframe[x_col].values
    labels = dataframe[y_col].values
    dataset = tf.data.Dataset.from_tensor_slices((filepaths, labels))
    dataset = dataset.map(preprocess_image, num_parallel_calls=AUTOTUNE)
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(dataframe))
    dataset = dataset.batch(batch_size).prefetch(AUTOTUNE)
    return dataset

# Create datasets
train_dataset = create_dataset(train_path, x_col="img_path", y_col="lbl", batch_size=BATCH_SIZE, shuffle=True)
val_dataset = create_dataset(val_path, x_col="img_path", y_col="lbl", batch_size=BATCH_SIZE, shuffle=False)
test_dataset = create_dataset(test_path, x_col="img_path", y_col="lbl", batch_size=BATCH_SIZE, shuffle=False)

# Debugging
for images, labels in train_dataset.take(1):
    print("Batch images shape:", images.shape)
    print("Batch labels shape:", labels.shape)

# --------------------------------------------------------------
# Define Model
# --------------------------------------------------------------
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten

img_shape = (224, 224, 3)
base_model = tf.keras.applications.Xception(include_top=False, weights="imagenet", input_shape=img_shape, pooling='max')

model = Sequential([
    base_model,
    Flatten(),
    Dropout(rate=0.3),
    Dense(128, activation='relu'),
    Dropout(rate=0.25),
    Dense(4, activation='softmax')  # Output layer for 4 classes
])

# Compile the model
model.compile(
    optimizer=tf.keras.optimizers.Adamax(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Print the model summary
# model.summary()

# --------------------------------------------------------------
# Train the Model
# --------------------------------------------------------------
history = model.fit(
    train_dataset,
    epochs=10,
    validation_data=val_dataset,
    validation_freq=1
)

# --------------------------------------------------------------
# Evaluate the Model
# --------------------------------------------------------------
test_loss, test_accuracy = model.evaluate(test_dataset)
print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy}")

