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


ep = 1
bs = 32

timing_results = []


def record_time(stage_name, start_time, end_time):
    elapsed_time = end_time - start_time
    timing_results.append({"Stage": stage_name, "Time (s)": elapsed_time})
    print(f"{stage_name} completed in {elapsed_time:.2f} seconds.")


MildDemented_dir = "/Volumes/JasonT7/2.Education/Research/Thesis/Paper/0017. alzheimerPrediction/data2/external/MildDemented"
ModerateDemented_dir = "/Volumes/JasonT7/2.Education/Research/Thesis/Paper/0017. alzheimerPrediction/data2/external/ModerateDemented"
NonDemented_dir = "/Volumes/JasonT7/2.Education/Research/Thesis/Paper/0017. alzheimerPrediction/data2/external/NonDemented"
VeryMildDemented_dir = "/Volumes/JasonT7/2.Education/Research/Thesis/Paper/0017. alzheimerPrediction/data2/external/VeryMildDemented"

filepaths, labels = [], []
cls_lbl = [
    "Mild Demented",
    "Moderate Demented",
    "Non Demented",
    "Very MildDemented",
]
dict_list = [
    MildDemented_dir,
    ModerateDemented_dir,
    NonDemented_dir,
    VeryMildDemented_dir,
]

for i, j in enumerate(dict_list):
    flist = os.listdir(j)
    for f in flist:
        fpath = os.path.join(j, f)
        filepaths.append(fpath)
        labels.append(cls_lbl[i])

start_time = time.time()

valid_filepaths, valid_labels = [], []
for filepath, label in zip(filepaths, labels):
    try:
        with Image.open(filepath) as img:
            img.verify()
            valid_filepaths.append(filepath)
            valid_labels.append(label)
    except (IOError, SyntaxError):
        print(f"Corrupted image file: {filepath}")

data_df = pd.DataFrame({"filepaths": valid_filepaths, "labels": valid_labels})
end_time = time.time()
record_time("Data Loading and Preprocessing", start_time, end_time)
print(data_df["labels"].value_counts())

from sklearn.model_selection import train_test_split

train_set, test_images = train_test_split(data_df, test_size=0.3, random_state=42)
val_set, test_images = train_test_split(test_images, test_size=0.5, random_state=42)

image_gen = ImageDataGenerator(preprocessing_function=preprocess_input)

train = image_gen.flow_from_dataframe(
    train_set,
    x_col="filepaths",
    y_col="labels",
    target_size=(224, 224),
    color_mode="rgb",
    class_mode="categorical",
    batch_size=bs,
    shuffle=False,
)

val = image_gen.flow_from_dataframe(
    val_set,
    x_col="filepaths",
    y_col="labels",
    target_size=(224, 224),
    color_mode="rgb",
    class_mode="categorical",
    batch_size=bs,
    shuffle=False,
)

test = image_gen.flow_from_dataframe(
    test_images,
    x_col="filepaths",
    y_col="labels",
    target_size=(224, 224),
    color_mode="rgb",
    class_mode="categorical",
    batch_size=bs,
    shuffle=False,
)

print(f"Train images:{len(train_set)}")
print(f"Val images:{len(val_set)}")
print(f"Test images:{len(test_images)}")

