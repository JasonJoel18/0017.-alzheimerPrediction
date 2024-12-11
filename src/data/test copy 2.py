import os
import pandas as pd
import numpy as np
import warnings
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.optimizers import Adamax
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

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

# Create DataFrame
data_df = pd.DataFrame({"filepaths": filepaths, "labels": labels})
print(data_df["labels"].value_counts())

# Train-test-validation split
from sklearn.model_selection import train_test_split
train_images, test_images = train_test_split(data_df, test_size=0.3, random_state=42)
train_set, val_set = train_test_split(train_images, test_size=0.2, random_state=42)

# Data augmentation and preprocessing
def normalize_images(image):
    return (image - np.mean(image)) / np.std(image)

image_gen = ImageDataGenerator(preprocessing_function=normalize_images)

train = image_gen.flow_from_dataframe(train_set, x_col="filepaths", y_col="labels",
                                    target_size=(224, 224), color_mode='rgb',
                                    class_mode="categorical", batch_size=32, shuffle=True)

val = image_gen.flow_from_dataframe(val_set, x_col="filepaths", y_col="labels",
                                    target_size=(224, 224), color_mode='rgb',
                                    class_mode="categorical", batch_size=32, shuffle=False)

test = image_gen.flow_from_dataframe(test_images, x_col="filepaths", y_col="labels",
                                    target_size=(224, 224), color_mode='rgb',
                                    class_mode="categorical", batch_size=32, shuffle=False)

# Define the model with EfficientNetB0
base_model = EfficientNetB0(include_top=False, weights="imagenet", input_shape=(224, 224, 3), pooling="avg")

model = Sequential([
    base_model,
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(4, activation='softmax')
])

model.compile(optimizer=Adamax(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy'])

# Add EarlyStopping and LearningRateScheduler
early_stopping = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
annealer = LearningRateScheduler(lambda x: 1e-3 * 0.95 ** x, verbose=0)

# Train the model
history = model.fit(train, epochs=2, validation_data=val, callbacks=[early_stopping, annealer])

# Evaluate the model
test_loss, test_accuracy = model.evaluate(test)
print(f"Test Accuracy: {test_accuracy:.2f}")

from tqdm import tqdm
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report, accuracy_score, recall_score, precision_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Monte Carlo Dropout for uncertainty estimation
def predict_with_uncertainty(model, dataset, n_samples=10):
    predictions = []

    # Loop over the dataset iterator with tqdm progress bar
    for _ in tqdm(range(n_samples), desc="Monte Carlo Sampling", dynamic_ncols=True):
        # Get a batch of images from the iterator
        images, _ = next(dataset)
        
        # Enable Dropout during inference
        preds = model(images, training=True)  # Enable Dropout during inference
        predictions.append(preds)
    
    predictions = tf.stack(predictions, axis=0)  # Shape: [n_samples, batch_size, num_classes]
    mean_preds = tf.reduce_mean(predictions, axis=0).numpy()  # Mean predictions
    uncertainty = tf.math.reduce_std(predictions, axis=0).numpy()  # Uncertainty
    return mean_preds, uncertainty

# Evaluate predictions on test set
mean_predictions, uncertainty = predict_with_uncertainty(model, test, n_samples=10)

# Get the true labels
y_true = test.classes

# Calculate classification metrics
y_pred = np.argmax(mean_predictions, axis=1)
report = classification_report(y_true, y_pred, target_names=class_labels, output_dict=True)

# Display metrics
class_metrics = pd.DataFrame(report).transpose()
print("\nPer-Class Metrics:")
print(class_metrics)

# --------------------------------------------------------------
# Metrics Calculation: Accuracy, Recall, Precision, F1-score
# --------------------------------------------------------------
accuracy = accuracy_score(y_true, y_pred)
recall = recall_score(y_true, y_pred, average=None)
precision = precision_score(y_true, y_pred, average=None)
f1 = f1_score(y_true, y_pred, average=None)

print(f"Accuracy: {accuracy:.2f}")
print(f"Recall (per class): {recall}")
print(f"Precision (per class): {precision}")
print(f"F1 Score (per class): {f1}")

# --------------------------------------------------------------
# Micro, Macro, and Weighted Averages
# --------------------------------------------------------------
micro_avg_recall = recall_score(y_true, y_pred, average='micro')
macro_avg_recall = recall_score(y_true, y_pred, average='macro')
weighted_avg_recall = recall_score(y_true, y_pred, average='weighted')

micro_avg_precision = precision_score(y_true, y_pred, average='micro')
macro_avg_precision = precision_score(y_true, y_pred, average='macro')
weighted_avg_precision = precision_score(y_true, y_pred, average='weighted')

micro_avg_f1 = f1_score(y_true, y_pred, average='micro')
macro_avg_f1 = f1_score(y_true, y_pred, average='macro')
weighted_avg_f1 = f1_score(y_true, y_pred, average='weighted')

print(f"Micro Average Recall: {micro_avg_recall:.2f}")
print(f"Macro Average Recall: {macro_avg_recall:.2f}")
print(f"Weighted Average Recall: {weighted_avg_recall:.2f}")

print(f"Micro Average Precision: {micro_avg_precision:.2f}")
print(f"Macro Average Precision: {macro_avg_precision:.2f}")
print(f"Weighted Average Precision: {weighted_avg_precision:.2f}")

print(f"Micro Average F1: {micro_avg_f1:.2f}")
print(f"Macro Average F1: {macro_avg_f1:.2f}")
print(f"Weighted Average F1: {weighted_avg_f1:.2f}")

# --------------------------------------------------------------
# Confusion Matrix with Uncertainty
# --------------------------------------------------------------
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_labels, yticklabels=class_labels)
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix")
plt.show()

# Highlight high-uncertainty predictions
threshold = 0.3  # Define uncertainty threshold
high_uncertainty_count = sum(u > threshold for u in uncertainty)
print(f"Number of high-uncertainty predictions (uncertainty > {threshold}): {high_uncertainty_count}")