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
MildDemented_dir = '/Volumes/Jason\'s T7/2. Education/Research/Thesis/Paper/0017. alzheimerPrediction/data/external/MildDemented'
ModerateDemented_dir = '/Volumes/Jason\'s T7/2. Education/Research/Thesis/Paper/0017. alzheimerPrediction/data/external/ModerateDemented'
NonDemented_dir = '/Volumes/Jason\'s T7/2. Education/Research/Thesis/Paper/0017. alzheimerPrediction/data/external/NonDemented'
VeryMildDemented_dir = '/Volumes/Jason\'s T7/2. Education/Research/Thesis/Paper/0017. alzheimerPrediction/data/external/VeryMildDemented'

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
train_images, test_images = train_test_split(data_df, test_size=0.3, random_state=42)
train_set, val_set = train_test_split(train_images, test_size=0.2, random_state=42)


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
                                    class_mode="categorical", batch_size=32, shuffle=True)

val = image_gen_val_test.flow_from_dataframe(val_set, x_col="filepaths", y_col="labels",
                                    target_size=(224, 224), color_mode='rgb',
                                    class_mode="categorical", batch_size=32, shuffle=False)

test = image_gen_val_test.flow_from_dataframe(test_images, x_col="filepaths", y_col="labels",
                                    target_size=(224, 224), color_mode='rgb',
                                    class_mode="categorical", batch_size=32, shuffle=False)


# def visualize_raw_images(images, actual_labels, class_labels):
#     plt.figure(figsize=(15, 10))
    
#     for i in range(10):
#         plt.subplot(2, 5, i + 1)
#         img = images[i]
        
#         # Assuming images are already in [0, 1] range or [0, 255], rescale if needed
#         img = np.clip(img, 0, 255).astype(np.uint8)  # Clipping to valid range and casting to uint8
        
#         plt.imshow(img)
#         plt.axis('off')
        
#         actual_label = class_labels[actual_labels[i]]
        
#         plt.title(f"Actual: {actual_label}", fontsize=10)
    
#     plt.tight_layout()
#     plt.show()

# Define the model with EfficientNetB0
base_model = EfficientNetB0(include_top=False, weights="imagenet", input_shape=(224, 224, 3), pooling="avg")

model = Sequential([
    base_model,
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(4, activation='softmax')
])

model.compile(optimizer=Adamax(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy'])

# Add EarlyStopping and LearningRateScheduler
early_stopping = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
# annealer = LearningRateScheduler(lambda x: 1e-3 * 0.95 ** x, verbose=0)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss', 
    factor=0.5, 
    patience=3, 
    min_lr=1e-6
)
# Train the model
history = model.fit(train, epochs=2, validation_data=val, callbacks=[early_stopping, reduce_lr])

# Evaluate the model
test_loss, test_accuracy = model.evaluate(test)
print(f"Test Accuracy: {test_accuracy:.2f}")

# Save the trained model
model.save('/Volumes/Jason\'s T7/2. Education/Research/Thesis/Paper/0017. alzheimerPrediction/models/jason_alzheimer_prediction_model.keras')
print("Model saved successfully.")

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



# Visualization function
def visualize_predictions_with_uncertainty(images, actual_labels, predicted_labels, uncertainty, class_labels):
    """
    Visualize 10 samples with actual label, predicted label, and uncertainty value.
    
    Args:
    - images: List of image tensors.
    - actual_labels: List of actual labels (indices).
    - predicted_labels: List of predicted labels (indices).
    - uncertainty: List of uncertainty values.
    - class_labels: List of class names.
    """
    plt.figure(figsize=(15, 10))
    
    for i in range(10):
        plt.subplot(2, 5, i + 1)
        img = images[i]
        
        # Assuming images are already in [0, 1] range or [0, 255], rescale if needed
        img = np.clip(img, 0, 255).astype(np.uint8)  # Clipping to valid range and casting to uint8
        
        plt.imshow(img)
        plt.axis('off')
        
        actual_label = class_labels[actual_labels[i]]
        predicted_label = class_labels[predicted_labels[i]]
        
        # Check if uncertainty[i] is an array and get the scalar value
        if isinstance(uncertainty[i], np.ndarray):
            uncertainty_value = uncertainty[i].flatten()[0]  # Extract the scalar value
        else:
            uncertainty_value = uncertainty[i]
        
        plt.title(f"Actual: {actual_label}\nPred: {predicted_label}\nUncertainty: {uncertainty_value:.2f}", fontsize=10)
    
    plt.tight_layout()
    plt.show()
    

# Extract 10 samples from the test set
sample_images, sample_labels = next(test)  # Get a batch from the test set
sample_images = sample_images[:10]  # Select the first 10 images

# Make predictions with uncertainty estimation
mean_preds, uncertainty = predict_with_uncertainty(model, test, n_samples=10)

# Get the predicted labels (index of highest probability class)
predicted_labels = np.argmax(mean_preds, axis=1)

# Get the actual labels (true values from the batch)
actual_labels = sample_labels[:10]
actual_labels_indices = np.argmax(actual_labels, axis=1)

# Visualize the predictions with uncertainty
visualize_predictions_with_uncertainty(sample_images, actual_labels_indices, predicted_labels, uncertainty[:10], class_labels)

# visualize_raw_images(sample_images, actual_labels_indices, predicted_labels)

# Save the trained model
model.save('/Volumes/Jason\'s T7/2. Education/Research/Thesis/Paper/0017. alzheimerPrediction/models/jason_alzheimer_prediction_model.keras')
print("Model saved successfully.")



import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors

def visualize_predictions_with_uncertainty(images, actual_labels, predicted_labels, uncertainty, class_labels):
    """
    Visualize 10 samples with actual label, predicted label, and confidence bar with indicator.
    
    Args:
    - images: List of image tensors.
    - actual_labels: List of actual labels (indices).
    - predicted_labels: List of predicted labels (indices).
    - uncertainty: List of uncertainty values (higher uncertainty means lower confidence).
    - class_labels: List of class names.
    """
    # Create a figure with 10 subplots (2 rows, 5 columns), more compact size
    fig, axes = plt.subplots(2, 5, figsize=(15, 8))  # Adjusted figure size to make it more compact
    plt.subplots_adjust(hspace=0.2, wspace=0.2)  # Reduced space between subplots
    
    for i in range(10):
        ax_img = axes[i // 5, i % 5]  # Get the appropriate axis for the image
        ax_bar = ax_img.inset_axes([0.0, -0.25, 1.0, 0.15])  # Make space for the confidence bar closer to image

        # Display the image
        img = images[i]
        img = np.clip(img, 0, 255).astype(np.uint8)  # Ensure valid range for images
        ax_img.imshow(img)
        ax_img.axis('off')  # Hide axes for better clarity of the image
        
        # Get the actual and predicted labels
        actual_label = class_labels[actual_labels[i]]
        predicted_label = class_labels[predicted_labels[i]]
        
        # Calculate uncertainty and confidence
        if isinstance(uncertainty[i], np.ndarray):
            uncertainty_value = uncertainty[i].flatten()[0]
        else:
            uncertainty_value = uncertainty[i]
        
        confidence = 1 - uncertainty_value  # Confidence is the inverse of uncertainty
        
        # Display title on top of the image
        ax_img.set_title(f"Actual: {actual_label}\nPred: {predicted_label}", fontsize=10, loc='center')
        
        # Create a more visible gradient bar for the confidence with more saturated colors
        cmap = mcolors.LinearSegmentedColormap.from_list("pastel_red_green", ['#FF4C4C', '#FFD34F', '#4CFF4C'])  # Red to Yellow to Green
        ax_bar.imshow(np.linspace(0, 1, 100).reshape(1, -1), aspect='auto', cmap=cmap)  # Gradient from red to green
        
        # Add a vertical line to show the confidence level on the bar
        ax_bar.plot([confidence * 100, confidence * 100], [0, 1], color='black', lw=2)  # Vertical line indicating confidence
        
        # Add the confidence value at the center of the bar
        ax_bar.text(50, 0.5, f'{confidence * 100:.2f}%', ha='center', va='center', color='black', fontsize=12, fontweight='bold')

        ax_bar.set_xlim(0, 100)  # Set limits for the bar to go from 0 to 100 (percentage scale)
        ax_bar.axis('off')  # Hide axes for the confidence bar

    # Final adjustments for compact spacing and clean layout
    plt.tight_layout()
    plt.show()
visualize_predictions_with_uncertainty(sample_images, actual_labels_indices, predicted_labels, uncertainty[:10], class_labels)
