import numpy as np
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.callbacks import EarlyStopping


# --------------------------------------------------------------
# Constants and Configuration
# --------------------------------------------------------------
img_shape = (224, 224,3)

base_model = tf.keras.applications.Xception(
    include_top=False, weights="imagenet", input_shape=img_shape, pooling='max')

model = Sequential([
    base_model,
    Flatten(),
    Dropout(rate=0.3),
    Dense(128, activation='relu'),
    Dropout(rate=0.25),
    Dense(4, activation='softmax')
])

model.compile(
    optimizer=tf.keras.optimizers.Adamax(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# --------------------------------------------------------------
# Train the Model
# --------------------------------------------------------------
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
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


# import sys

# # import tensorflow.keras
# import pandas as pd
# import sklearn as sk
# import scipy as sp
# import tensorflow as tf
# import platform

# print(f"Python Platform: {platform.platform()}")
# print(f"Tensor Flow Version: {tf.__version__}")
# # print(f"Keras Version: {tf.keras.__version__}")
# print()
# print(f"Python {sys.version}")
# print(f"Pandas {pd.__version__}")
# print(f"Scikit-Learn {sk.__version__}")
# print(f"SciPy {sp.__version__}")
# gpu = len(tf.config.list_physical_devices('GPU'))>0
# print("GPU is", "available" if gpu else "NOT AVAILABLE")

