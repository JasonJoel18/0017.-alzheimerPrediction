
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
test_loss, test_accuracy = model.evaluate(test)
end_time = time.time()
record_time("Model Evaluation", start_time, end_time)
print(f"Test Accuracy: {test_accuracy}")

num_train_images = len(train_set)
model_save_path = f"/Volumes/JasonT7/2.Education/Research/Thesis/Paper/0017. alzheimerPrediction/models/jason_alzheimer_prediction_model_{len(train_set)}_images_{ep}_epochs.keras"
model.save(model_save_path)
print("Model saved successfully.")


model = tf.keras.models.load_model(
    f"/Volumes/JasonT7/2.Education/Research/Thesis/Paper/0017. alzheimerPrediction/models/jason_alzheimer_prediction_model_{len(train_set)}_images_{ep}_epochs.keras"
)

from tqdm import tqdm
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    recall_score,
    precision_score,
    f1_score,
    confusion_matrix,
)
import seaborn as sns
import matplotlib.pyplot as plt


def predict_with_uncertainty(model, dataset, n_samples=50):
    predictions = []
    total_batches = len(dataset)
    pbar_outer = tqdm(total=n_samples, desc="Monte Carlo Sampling", dynamic_ncols=True)

    for sample_idx in range(n_samples):
        batch_preds = []
        dataset.reset()
        pbar_inner = tqdm(
            total=total_batches,
            position=1,
            leave=False,
            desc=f"Processing Sample {sample_idx + 1}/{n_samples}",
            ncols=80,
        )

        for batch_idx, (images, _) in enumerate(dataset):
            if batch_idx >= total_batches:
                break
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


start_time = time.time()
mean_predictions, uncertainty = predict_with_uncertainty(model, test, n_samples=5)
end_time = time.time()
record_time("Prediction", start_time, end_time)

y_true = test.classes
y_true = np.array(y_true)
y_pred = np.argmax(mean_predictions, axis=1)

cls_ct = np.bincount(y_true)
cls_wt = cls_ct / len(y_true)

report = classification_report(
    y_true, y_pred, target_names=cls_lbl, output_dict=True
)

m_df = pd.DataFrame()

for no, cls in enumerate(cls_lbl):
    m_df.loc[cls, "precision"] = report[cls]["precision"]
    m_df.loc[cls, "recall"] = report[cls]["recall"]
    m_df.loc[cls, "f1-score"] = report[cls]["f1-score"]

    cls_msk = y_true == no
    class_accuracy = np.mean(y_pred[cls_msk] == y_true[cls_msk])
    m_df.loc[cls, "accuracy"] = class_accuracy

m_df.loc["macro avg", "precision"] = m_df.loc[
    cls_lbl, "precision"
].mean()
m_df.loc["macro avg", "recall"] = m_df.loc[cls_lbl, "recall"].mean()
m_df.loc["macro avg", "f1-score"] = m_df.loc[
    cls_lbl, "f1-score"
].mean()
m_df.loc["macro avg", "accuracy"] = m_df.loc[
    cls_lbl, "accuracy"
].mean()

m_df.loc["weighted avg", "precision"] = np.average(
    m_df.loc[cls_lbl, "precision"], weights=cls_wt
)
m_df.loc["weighted avg", "recall"] = np.average(
    m_df.loc[cls_lbl, "recall"], weights=cls_wt
)
m_df.loc["weighted avg", "f1-score"] = np.average(
    m_df.loc[cls_lbl, "f1-score"], weights=cls_wt
)
m_df.loc["weighted avg", "accuracy"] = np.average(
    m_df.loc[cls_lbl, "accuracy"], weights=cls_wt
)

m_df = m_df.round(2)

print("\nClassification Metrics by Class:")
display(m_df)

threshold = 0.2
high_uncertainty_count = sum(u > threshold for u in uncertainty)
print(
    f"No. high-uncertainty pred where (uncertainty > {threshold}): {high_uncertainty_count}"
)

sample_images, sample_labels = next(test)
sample_images = sample_images[:10]
mean_preds, uncertainty = predict_with_uncertainty(model, test, n_samples=10)
predicted_labels = np.argmax(mean_preds, axis=1)
actual_labels = sample_labels[:10]
actual_labels_indices = np.argmax(actual_labels, axis=1)
