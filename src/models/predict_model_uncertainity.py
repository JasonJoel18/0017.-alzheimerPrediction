import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# --------------------------------------------------------------
# Constants and Configuration
# --------------------------------------------------------------
img_shape = (244, 244, 3)
BATCH_SIZE = 32
DROPOUT_RATE = 0.3
N_SAMPLES = 50  # Number of Monte Carlo samples during inference

class_names = ['Mild Demented', 'Moderate Demented', 'Non Demented', 'Very Mild Demented']

# --------------------------------------------------------------
# Modified Model with Monte Carlo Dropout
# --------------------------------------------------------------
class MCDropoutModel(tf.keras.Model):
    def __init__(self, base_model):
        super(MCDropoutModel, self).__init__()
        self.base_model = base_model
        self.dropout_1 = Dropout(DROPOUT_RATE)
        self.fc_1 = Dense(256, activation="relu")
        self.dropout_2 = Dropout(DROPOUT_RATE)
        self.fc_2 = Dense(len(class_names), activation="softmax")

    def call(self, inputs, training=False):
        x = self.base_model(inputs, training=training)
        x = Flatten()(x)
        x = self.dropout_1(x, training=training)
        x = self.fc_1(x)
        x = self.dropout_2(x, training=training)
        return self.fc_2(x)

# Base model (transfer learning backbone)
base_model = tf.keras.applications.Xception(include_top=False, weights="imagenet", input_shape=img_shape, pooling='max')

# Build Monte Carlo model
mc_model = MCDropoutModel(base_model)
mc_model.compile(optimizer=tf.keras.optimizers.Adamax(learning_rate=0.001),
                 loss='categorical_crossentropy', metrics=['accuracy'])

# --------------------------------------------------------------
# Train the Model
# --------------------------------------------------------------
history = mc_model.fit(
    train_dataset,
    epochs=1,  # Use more epochs for final training
    validation_data=val_dataset,
    validation_freq=1
)

# --------------------------------------------------------------
# Monte Carlo Sampling for Uncertainty Estimation
# --------------------------------------------------------------
def monte_carlo_predictions(mc_model, dataset, n_samples):
    """
    Perform Monte Carlo sampling for uncertainty estimation.
    Args:
        mc_model: Model with Monte Carlo Dropout enabled.
        dataset: Dataset to evaluate on.
        n_samples: Number of Monte Carlo samples.

    Returns:
        mean_preds: Mean predictions across samples.
        uncertainty: Standard deviation of predictions across samples.
    """
    predictions = []

    for i in range(n_samples):
        preds = []
        for data, _ in dataset:  # Iterate over the dataset
            preds.append(mc_model(data, training=True))  # Use the call method
        preds = tf.concat(preds, axis=0)  # Concatenate batch predictions
        predictions.append(preds)

    predictions = tf.stack(predictions, axis=0)  # Stack all predictions
    mean_preds = tf.reduce_mean(predictions, axis=0)  # Mean prediction
    uncertainty = tf.math.reduce_std(predictions, axis=0)  # Uncertainty (stddev)
    return mean_preds.numpy(), uncertainty.numpy()

# --------------------------------------------------------------
# Evaluate Model with Uncertainty
# --------------------------------------------------------------
def evaluate_with_uncertainty(mc_model, test_dataset, n_samples):
    """
    Evaluate the model on the test dataset and compute uncertainty.
    Args:
        mc_model: Model with Monte Carlo Dropout enabled.
        test_dataset: Test dataset to evaluate on.
        n_samples: Number of Monte Carlo samples.

    Returns:
        mean_preds: Mean predictions.
        uncertainty: Uncertainty (stddev) of predictions.
    """
    mean_preds, uncertainty = monte_carlo_predictions(mc_model, test_dataset, n_samples)
    true_labels = []
    for _, labels in test_dataset:
        true_labels.append(labels)
    true_labels = tf.concat(true_labels, axis=0).numpy()

    predicted_classes = np.argmax(mean_preds, axis=1)
    uncertainty_values = uncertainty.max(axis=1)

    print("Classification Report:")
    print(classification_report(true_labels, predicted_classes, target_names=class_names))
    
    # Confusion Matrix
    cm = confusion_matrix(true_labels, predicted_classes)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()
    
    # Display uncertainty distribution
    plt.hist(uncertainty_values, bins=20, color='skyblue', edgecolor='black')
    plt.title("Uncertainty Distribution")
    plt.xlabel("Uncertainty")
    plt.ylabel("Frequency")
    plt.show()

    return mean_preds, uncertainty

# Evaluate model
mean_preds, uncertainty = evaluate_with_uncertainty(mc_model, test_dataset, N_SAMPLES)

# --------------------------------------------------------------
# Visualize Predictions with Uncertainty
# --------------------------------------------------------------
def visualize_predictions_with_uncertainty(test_dataset, mean_preds, uncertainty, class_names, n_images=10):
    """
    Visualize model predictions and uncertainties.
    """
    for images, labels in test_dataset.take(1):
        true_labels = np.argmax(labels, axis=1)
        predicted_classes = np.argmax(mean_preds, axis=1)
        uncertainties = uncertainty.max(axis=1)

        plt.figure(figsize=(15, 10))
        for i in range(n_images):
            plt.subplot(2, 5, i + 1)
            img = (images[i].numpy() + 1) / 2.0  # Scale images to [0, 1]
            plt.imshow(img)
            
            true_label = class_names[true_labels[i]]
            pred_label = class_names[predicted_classes[i]]
            unc = uncertainties[i]

            color = "green" if true_label == pred_label else "red"
            plt.title(f"True: {true_label}\nPred: {pred_label}\nUnc: {unc:.2f}", color=color)
            plt.axis('off')
        plt.tight_layout()
        plt.show()

visualize_predictions_with_uncertainty(test_dataset, mean_preds, uncertainty, class_names)