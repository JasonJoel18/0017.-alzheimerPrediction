
import os
import pandas as pd
import numpy as np
import warnings
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import EfficientNetB0, preprocess_input
from tensorflow.keras.optimizers import Adamax
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from PIL import Image
from tensorflow.keras.callbacks import ReduceLROnPlateau
from IPython.display import display
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors
from rich import print
import time

warnings.filterwarnings(action="ignore")

start_time = time.time()

base_model = EfficientNetB0(
    include_top=False, weights="imagenet", input_shape=(224, 224, 3), pooling="avg"
)

model = Sequential(
    [
        base_model,
        Flatten(),
        Dense(128, activation="relu"),
        Dropout(0.3),
        Dense(4, activation="softmax"),
    ]
)

model.compile(
    optimizer=Adamax(learning_rate=0.001),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

early_stopping = EarlyStopping(
    monitor="val_loss", patience=5, restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6)
end_time = time.time()
record_time("Model Building", start_time, end_time)

model.summary()


start_time = time.time()
history = model.fit(
    train, epochs=ep, validation_data=val, callbacks=[early_stopping, reduce_lr]
)
end_time = time.time()
record_time("Model Training", start_time, end_time)

model = tf.keras.models.load_model(
    f"/Volumes/JasonT7/2.Education/Research/Thesis/Paper/0017. alzheimerPrediction/models/jason_alzheimer_prediction_model_{len(train_set)}_images_{ep}_epochs.keras"
)