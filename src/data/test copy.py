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

warnings.filterwarnings(action="ignore")

# Paths to the dataset
MildDemented_dir = '/Volumes/Jason\'s T7/2. Education/Research/Thesis/Paper/0017. alzheimerPrediction/data2/external/MildDemented'
ModerateDemented_dir = '/Volumes/Jason\'s T7/2. Education/Research/Thesis/Paper/0017. alzheimerPrediction/data2/external/ModerateDemented'
NonDemented_dir = '/Volumes/Jason\'s T7/2. Education/Research/Thesis/Paper/0017. alzheimerPrediction/data2/external/NonDemented'
VeryMildDemented_dir = '/Volumes/Jason\'s T7/2. Education/Research/Thesis/Paper/0017. alzheimerPrediction/data2/external/VeryMildDemented'

# Load dataset
filepaths, labels = [], []
class_labels = ['Mild Demented', 'Moderate Demented', 'Non Demented', 'Very MildDemented']
dict_list = [MildDemented_dir, ModerateDemented_dir, NonDemented_dir, VeryMildDemented_dir]

for i, j in enumerate(dict_list):
    flist = os.listdir(j)
    for f in flist:
        fpath = os.path.join(j, f)
        filepaths.append(fpath)
        labels.append(class_labels[i])

# Filter out corrupted image files
valid_filepaths, valid_labels = [], []
for filepath, label in zip(filepaths, labels):
    try:
        with Image.open(filepath) as img:
            img.verify()  # Verify that the file is not corrupted
            valid_filepaths.append(filepath)
            valid_labels.append(label)
    except (IOError, SyntaxError):
        print(f"Corrupted image file: {filepath}")
        
# Create DataFrame
data_df = pd.DataFrame({"filepaths": valid_filepaths, "labels": valid_labels})
print(data_df["labels"].value_counts())


# Train-test-validation split
from sklearn.model_selection import train_test_split
train_set, test_images = train_test_split(data_df, test_size=0.3, random_state=42)
val_set, test_images  = train_test_split(test_images, test_size=0.5, random_state=42)


image_gen_train = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)
image_gen_val_test = ImageDataGenerator(
    preprocessing_function=preprocess_input
)


train = image_gen_train.flow_from_dataframe(train_set, x_col="filepaths", y_col="labels",
                                    target_size=(224, 224), color_mode='rgb',
                                    class_mode="categorical", batch_size=6)

val = image_gen_val_test.flow_from_dataframe(val_set, x_col="filepaths", y_col="labels",
                                    target_size=(224, 224), color_mode='rgb',
                                    class_mode="categorical", batch_size=6)

test = image_gen_val_test.flow_from_dataframe(test_images, x_col="filepaths", y_col="labels",
                                    target_size=(224, 224), color_mode='rgb',
                                    class_mode="categorical", batch_size=6)

print(f'Train images:{len(train_set)}')
print(f'Val images:{len(val_set)}')
print(f'Test images:{len(test_images)}')

from tqdm import tqdm
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report, accuracy_score, recall_score, precision_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

model = tf.keras.models.load_model('/Volumes/Jason\'s T7/2. Education/Research/Thesis/Paper/0017. alzheimerPrediction/models/jason_alzheimer_prediction_model.keras')

import numpy as np
from tqdm import tqdm

# Monte Carlo Dropout for uncertainty estimation
def predict_with_uncertainty(model, dataset, n_samples=1):
    predictions = []
    total_batches = len(dataset)
    pbar_outer = tqdm(total=n_samples, desc="Monte Carlo Sampling", dynamic_ncols=True)

    for sample_idx in range(n_samples):
        batch_preds = []
        dataset.reset()
        pbar_inner = tqdm(total=total_batches, position=1, leave=False, desc=f"Processing Sample {sample_idx + 1}/{n_samples}", ncols=80)

        for batch_idx, (images, _) in enumerate(dataset):
            if batch_idx >= total_batches:
                break

            # print(f"Processing batch {batch_idx + 1}/{total_batches}")
            preds = model(images, training=True)
            batch_preds.append(preds.numpy())
            pbar_inner.update(1)
        pbar_inner.close()
        predictions.append(np.vstack(batch_preds))
        pbar_outer.update(1)
    pbar_outer.close()
    predictions = np.stack(predictions, axis=0)
    mean_preds = np.mean(predictions, axis=0)
    uncertainty = np.std(predictions, axis=0)

    return mean_preds, uncertainty

# Evaluate the model with Monte Carlo Dropout on the test dataset
mean_predictions, uncertainty = predict_with_uncertainty(model, test, n_samples=1)


# Get the true labels
y_true = test.classes

# Ensure y_pred aligns with the entire test set
y_pred = np.argmax(mean_predictions, axis=1)


y_true_test = np.array(y_true) 
print(y_pred.shape)
print(y_true_test.shape)

assert len(y_true) == len(y_pred), "Mismatch between true labels and predictions."

# Generate classification report
report = classification_report(y_true, y_pred, target_names=class_labels, output_dict=True)



# Display per-class metrics
class_metrics = pd.DataFrame(report).transpose()
print("\nPer-Class Metrics:")
print(class_metrics)
