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

ep = 2
bs = 32

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
                                    class_mode="categorical", batch_size=bs, shuffle=True)

val = image_gen_val_test.flow_from_dataframe(val_set, x_col="filepaths", y_col="labels",
                                    target_size=(224, 224), color_mode='rgb',
                                    class_mode="categorical", batch_size=bs, shuffle=False)

test = image_gen_val_test.flow_from_dataframe(test_images, x_col="filepaths", y_col="labels",
                                    target_size=(224, 224), color_mode='rgb',
                                    class_mode="categorical", batch_size=bs, shuffle=False)

print(f'Train images:{len(train_set)}')
print(f'Val images:{len(val_set)}')
print(f'Test images:{len(test_images)}')
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
history = model.fit(train, epochs=ep, validation_data=val, callbacks=[early_stopping, reduce_lr])

# Evaluate the model
test_loss, test_accuracy = model.evaluate(test)
print(f"Test Accuracy: {test_accuracy:.2f}")

# Save the trained model
num_train_images = len(train_set)
model_save_path = f'/Volumes/Jason\'s T7/2. Education/Research/Thesis/Paper/0017. alzheimerPrediction/models/jason_alzheimer_prediction_model_{num_train_images}_images_{ep}_epochs.keras'
model.save(model_save_path)
print("Model saved successfully.")


# =================================================================
# =================================================================
# =================================================================
# =================================================================
# =================================================================


model = tf.keras.models.load_model('/Volumes/Jason\'s T7/2. Education/Research/Thesis/Paper/0017. alzheimerPrediction/models/jason_alzheimer_prediction_model.keras')

from tqdm import tqdm
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report, accuracy_score, recall_score, precision_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Monte Carlo Dropout for uncertainty estimation
def predict_with_uncertainty(model, dataset, n_samples=5):
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

# Evaluate predictions on test set
mean_predictions, uncertainty = predict_with_uncertainty(model, test, n_samples=5)

# Get the true labels
y_true = test.classes

y_true = np.array(y_true)  # Ensure y_true is a numpy array if it's not already
y_pred = np.argmax(mean_predictions, axis=1)  # Assuming mean_predictions is the model's output

# Generate the classification report
report = classification_report(y_true, y_pred, target_names=class_labels, output_dict=True)

# Convert the classification report to a DataFrame
class_metrics = pd.DataFrame(report).transpose()

# Calculate per-class accuracy
class_accuracy = [np.sum(y_pred == idx) / len(y_true) for idx in range(len(class_labels))]

# Add per-class accuracy to the DataFrame
class_metrics.loc[class_labels, 'accuracy'] = class_accuracy

# Calculate overall accuracy for macro, micro, and weighted averages
accuracy_macro = np.mean(class_accuracy)  # Average of per-class accuracy
accuracy_micro = np.sum(y_pred == y_true) / len(y_true)  # Micro average (total correct predictions)
accuracy_weighted = np.average(class_accuracy, weights=[np.sum(y_true == i) for i in range(len(class_labels))])  # Weighted average

# Update the averages in the DataFrame
class_metrics.loc['macro avg', 'accuracy'] = accuracy_macro
class_metrics.loc['micro avg', 'accuracy'] = accuracy_micro
class_metrics.loc['weighted avg', 'accuracy'] = accuracy_weighted

# Micro average for precision, recall, and f1-score
precision_micro = precision_score(y_true, y_pred, average='micro')
recall_micro = recall_score(y_true, y_pred, average='micro')
f1_micro = f1_score(y_true, y_pred, average='micro')

# Update the DataFrame with micro averages for precision, recall, and f1-score
class_metrics.loc['micro avg', 'precision'] = precision_micro
class_metrics.loc['micro avg', 'recall'] = recall_micro
class_metrics.loc['micro avg', 'f1-score'] = f1_micro

# Display the updated DataFrame
print("\nPer-Class Metrics with Micro Averages:")
class_metrics


# --------------------------------------------------------------
# Confusion Matrix with Uncertainty
# --------------------------------------------------------------


# Highlight high-uncertainty predictions
threshold = 0.3  # Define uncertainty threshold
high_uncertainty_count = sum(u > threshold for u in uncertainty)
print(f"Number of high-uncertainty predictions (uncertainty > {threshold}): {high_uncertainty_count}")
    

sample_images, sample_labels = next(test)
sample_images = sample_images[:10]
mean_preds, uncertainty = predict_with_uncertainty(model, test, n_samples=10)
predicted_labels = np.argmax(mean_preds, axis=1)
actual_labels = sample_labels[:10]
actual_labels_indices = np.argmax(actual_labels, axis=1)


import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors


def visualize_predictions_with_uncertainty(images, actual_labels, predicted_labels, uncertainty, class_labels):
    # Create a figure with 10 subplots (2 rows, 5 columns), more compact size
    fig, axes = plt.subplots(2, 5, figsize=(15, 8))  # Adjusted figure size to make it more compact
    plt.subplots_adjust(hspace=0.5, wspace=0.2)  # Adjust space between subplots for better clarity
    
    for i in range(10):
        ax_img = axes[i // 5, i % 5]  # Get the appropriate axis for the image
        ax_bar = ax_img.inset_axes([0.0, -0.25, 1.0, 0.1])  # Make space for the confidence bar closer to image
        
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
        
        # Add the confidence value just below the image, centered
        ax_img.text(0.5, -0.1, f'{confidence * 100:.2f}%', ha='center', va='center', color='black', fontsize=12, fontweight='bold', transform=ax_img.transAxes)

        ax_bar.set_xlim(0, 100)  # Set limits for the bar to go from 0 to 100 (percentage scale)
        ax_bar.axis('off')  # Hide axes for the confidence bar

    # Final adjustments for compact spacing and clean layout
    plt.tight_layout()
    plt.show()

visualize_predictions_with_uncertainty(sample_images, actual_labels_indices, predicted_labels, uncertainty[:10], class_labels)
