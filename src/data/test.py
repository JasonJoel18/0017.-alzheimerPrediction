import os
import pandas as pd
import tensorflow as tf
from glob import glob
from sklearn.model_selection import train_test_split
import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.callbacks import EarlyStopping

print("TensorFlow version:", tf.__version__)

# --------------------------------------------------------------
# Constants and Configuration
# --------------------------------------------------------------
IMAGE_SIZE = (244, 244)
BATCH_SIZE = 32
AUTOTUNE = tf.data.AUTOTUNE
EPOCHS = 10

# Class directories
data_dirs = {
    'Mild Demented': '/Volumes/Jason\'s T7/2. Education/Research/Thesis/Paper/0017. alzheimerPrediction/data2/external/MildDemented',
    'Moderate Demented': '/Volumes/Jason\'s T7/2. Education/Research/Thesis/Paper/0017. alzheimerPrediction/data2/external/ModerateDemented',
    'Non Demented': '/Volumes/Jason\'s T7/2. Education/Research/Thesis/Paper/0017. alzheimerPrediction/data2/external/NonDemented',
    'Very MildDemented': '/Volumes/Jason\'s T7/2. Education/Research/Thesis/Paper/0017. alzheimerPrediction/data2/external/VeryMildDemented'
}

label_mapping = {'Non Demented': 0, 'Moderate Demented': 1,
                 'Mild Demented': 2, 'Very MildDemented': 3}
class_names = list(label_mapping.keys())

# --------------------------------------------------------------
# Data Preparation
# --------------------------------------------------------------
# Collect image paths and labels
img_paths = []
labels = []
for cls, dir_path in data_dirs.items():
    for img in os.listdir(dir_path):
        img_paths.append(os.path.join(dir_path, img))
        labels.append(cls)

# Create DataFrame
df = pd.DataFrame({'img_path': img_paths, 'lbl': labels})
df['lbl'] = df['lbl'].map(label_mapping)

# Train-test split
train_data, temp_data = train_test_split(df, test_size=0.4, random_state=42)
val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)

# --------------------------------------------------------------
# Dataset Preparation
# --------------------------------------------------------------
def preprocess_image(filepath, label):
    def _load_image(filepath):
        filepath = filepath.decode('utf-8')
        image = cv2.imread(filepath)
        image = cv2.resize(image, IMAGE_SIZE)
        return tf.keras.applications.mobilenet_v2.preprocess_input(image)

    image = tf.numpy_function(_load_image, [filepath], tf.float32)
    image.set_shape(IMAGE_SIZE + (3,))
    label = tf.one_hot(label, depth=4)
    return image, label

def create_dataset(df, x_col, y_col, batch_size, shuffle=False):
    df = df[df[x_col].apply(os.path.exists)]
    filepaths = df[x_col].values
    labels = df[y_col].values
    dataset = tf.data.Dataset.from_tensor_slices((filepaths, labels))
    dataset = dataset.map(preprocess_image, num_parallel_calls=AUTOTUNE)
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(df))
    return dataset.batch(batch_size).prefetch(AUTOTUNE)

train_dataset = create_dataset(train_data, 'img_path', 'lbl', BATCH_SIZE, shuffle=True)
val_dataset = create_dataset(val_data, 'img_path', 'lbl', BATCH_SIZE)
test_dataset = create_dataset(test_data, 'img_path', 'lbl', BATCH_SIZE)

# --------------------------------------------------------------
# Model with Monte Carlo Dropout
# --------------------------------------------------------------
img_shape = IMAGE_SIZE + (3,)

base_model = tf.keras.applications.Xception(
    include_top=False, weights='imagenet', input_shape=img_shape, pooling='avg')

model = Sequential([
    base_model,
    Dropout(0.3),  # Monte Carlo Dropout
    Dense(128, activation='relu'),
    Dropout(0.3),  # Monte Carlo Dropout
    Dense(4, activation='softmax')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# --------------------------------------------------------------
# Training the Model
# --------------------------------------------------------------
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
history = model.fit(
    train_dataset,
    epochs=EPOCHS,
    validation_data=val_dataset,
    callbacks=[early_stopping]
)

# --------------------------------------------------------------
# Monte Carlo Sampling for Uncertainty Estimation
# --------------------------------------------------------------
def monte_carlo_predictions(mc_model, dataset, n_samples=10):
    """
    Perform Monte Carlo sampling for uncertainty estimation.
    Args:
        mc_model: Trained model with Dropout enabled at inference.
        dataset: Dataset to evaluate.
        n_samples: Number of Monte Carlo samples.

    Returns:
        mean_preds: Mean predictions across samples.
        uncertainty: Uncertainty (standard deviation) across samples.
    """
    predictions = []
    for _ in range(n_samples):
        preds = []
        for images, _ in dataset:
            preds.append(mc_model(images, training=True))  # Enable dropout
        predictions.append(tf.concat(preds, axis=0))
    predictions = tf.stack(predictions, axis=0)
    mean_preds = tf.reduce_mean(predictions, axis=0)
    uncertainty = tf.math.reduce_std(predictions, axis=0)
    return mean_preds.numpy(), uncertainty.numpy()

# --------------------------------------------------------------
# Evaluation with Uncertainty
# --------------------------------------------------------------
mean_preds, uncertainty = monte_carlo_predictions(model, test_dataset, n_samples=10)
true_labels = np.concatenate([np.argmax(y, axis=1) for _, y in test_dataset], axis=0)
predicted_labels = np.argmax(mean_preds, axis=1)

# Classification report
report = classification_report(true_labels, predicted_labels, target_names=class_names)
print(report)

# Confusion Matrix
cm = confusion_matrix(true_labels, predicted_labels)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()

# --------------------------------------------------------------
# Visualization of Uncertain Predictions
# --------------------------------------------------------------
def visualize_uncertainty(images, true_labels, pred_labels, uncertainty, class_names, num_images=10):
    indices = np.argsort(uncertainty)[::-1][:num_images]  # Select most uncertain samples
    plt.figure(figsize=(15, 10))
    for i, idx in enumerate(indices):
        plt.subplot(2, 5, i + 1)
        plt.imshow((images[idx] + 1) / 2.0)  # Rescale from [-1, 1] to [0, 1]
        plt.title(f"True: {class_names[true_labels[idx]]}\nPred: {class_names[pred_labels[idx]]}\nUnc: {uncertainty[idx]:.3f}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()

# Visualize predictions with highest uncertainty
images, _ = next(iter(test_dataset.unbatch().batch(len(test_data))))
visualize_uncertainty(images.numpy(), true_labels, predicted_labels, uncertainty.max(axis=1), class_names)