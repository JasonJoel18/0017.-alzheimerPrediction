import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import seaborn as sns

class_names = ['Mild Demented', 'Moderate Demented', 'Non Demented', 'Very MildDemented']

# --------------------------------------------------------------
# Uncertainty-Aware Model Evaluation
# --------------------------------------------------------------
def evaluate_model_with_uncertainty(mc_model, dataset, class_names, n_samples=10):
    """
    Evaluate the model with uncertainty estimation using Monte Carlo Dropout.

    Args:
    - mc_model: Trained TensorFlow model with Dropout enabled.
    - dataset: tf.data.Dataset to evaluate.
    - class_names: List of class names.
    - n_samples: Number of Monte Carlo samples for uncertainty estimation.

    Returns:
    - metrics_df: DataFrame containing per-class metrics.
    - avg_metrics: Dictionary of micro, macro, and weighted averages.
    - true_labels: Ground truth labels.
    - mean_predictions: Mean predictions across Monte Carlo samples.
    - uncertainty: Uncertainty (standard deviation) across Monte Carlo samples.
    """
    true_labels = []
    mean_predictions = []
    uncertainty_list = []

    for images, labels in dataset:
        # Monte Carlo Sampling
        predictions = []
        for _ in range(n_samples):
            preds = mc_model(images, training=True)  # Enable Dropout
            predictions.append(preds)
        
        predictions = tf.stack(predictions, axis=0)  # Shape: [n_samples, batch_size, num_classes]
        mean_preds = tf.reduce_mean(predictions, axis=0).numpy()  # Mean predictions
        uncertainty = tf.math.reduce_std(predictions, axis=0).numpy()  # Uncertainty
        
        mean_predictions.extend(np.argmax(mean_preds, axis=1))  # Predicted labels
        uncertainty_list.extend(np.max(uncertainty, axis=1))  # Max uncertainty per prediction
        true_labels.extend(np.argmax(labels, axis=1))  # True labels

    # Classification Report
    report = classification_report(
        true_labels,
        mean_predictions,
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )
    
    metrics_df = pd.DataFrame(report).transpose()
    class_metrics = metrics_df.iloc[:-3, :]  # Exclude averages
    avg_metrics = metrics_df.iloc[-3:, :]  # Micro, macro, weighted averages

    return class_metrics, avg_metrics, true_labels, mean_predictions, uncertainty_list

# Evaluate the model
class_metrics, avg_metrics, true_labels, mean_predictions, uncertainties = evaluate_model_with_uncertainty(
    model, test_dataset, class_names, n_samples=10
)

# Display metrics
print("\nPer-Class Metrics:")
print(class_metrics.to_string(index=True))
print("\nAverage Metrics:")
print(avg_metrics.to_string(index=True))

# --------------------------------------------------------------
# Confusion Matrix with Uncertainty
# --------------------------------------------------------------
def plot_uncertainty_confusion_matrix(true_labels, predictions, uncertainties, class_names, threshold=0.3):
    """
    Plot a confusion matrix highlighting uncertain predictions.

    Args:
    - true_labels: Ground truth labels.
    - predictions: Predicted labels.
    - uncertainties: List of uncertainty values for predictions.
    - class_names: List of class names.
    - threshold: Uncertainty threshold to flag high-uncertainty predictions.
    """
    cm = confusion_matrix(true_labels, predictions)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title("Confusion Matrix")
    plt.show()

    # Highlight high-uncertainty predictions
    high_uncertainty_count = sum(u > threshold for u in uncertainties)
    print(f"Number of high-uncertainty predictions (uncertainty > {threshold}): {high_uncertainty_count}")

plot_uncertainty_confusion_matrix(true_labels, mean_predictions, uncertainties, class_names)

# --------------------------------------------------------------
# Visualization with Uncertainty
# --------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

def visualize_predictions_with_uncertainty(mc_model, dataset, class_names, uncertainties, num_images=10):
    """
    Visualize predictions, highlighting uncertain ones and adding color indicators for uncertainty.

    Args:
    - mc_model: Trained TensorFlow model with Dropout enabled.
    - dataset: tf.data.Dataset to evaluate.
    - class_names: List of class names.
    - uncertainties: List of uncertainty values for predictions.
    - num_images: Number of images to display.
    """
    uncertainties = np.array(uncertainties)  # Ensure uncertainties are in numpy format
    
    dataset = dataset.unbatch()
    dataset = dataset.batch(num_images)

    for images, labels in dataset.take(1):  # Take one batch of data
        preds = []
        for _ in range(10):  # Monte Carlo sampling
            preds.append(mc_model(images, training=True))  # Enable dropout

        preds = tf.reduce_mean(tf.stack(preds, axis=0), axis=0).numpy()
        predicted_labels = np.argmax(preds, axis=1)
        true_labels = np.argmax(labels.numpy(), axis=1)

        images = (images + 1) / 2.0  # Scale images to [0, 1]

        plt.figure(figsize=(15, 10))
        for i in range(min(num_images, len(images))):
            plt.subplot(2, 5, i + 1)
            img = images[i].numpy()
            plt.imshow(img)

            true_label = class_names[true_labels[i]]
            pred_label = class_names[predicted_labels[i]]
            uncertainty = uncertainties[i]  # Match uncertainty to this index

            # Determine the color based on uncertainty
            if uncertainty < 0.2:
                indicator_color = "green"  # 🟢
            elif 0.2 <= uncertainty < 0.4:
                indicator_color = "yellow"  # 🟡
            else:
                indicator_color = "red"  # 🔴

            # Set color based on correctness
            title_color = "green" if true_label == pred_label else "red"
            plt.title(f"True: {true_label}\nPred: {pred_label}\nUnc: {uncertainty:.3f}", color=title_color)
            plt.axis('off')

            # Add a colored dot as uncertainty indicator
            plt.scatter(0.05, 0.05, s=100, c=indicator_color, transform=plt.gca().transAxes, marker='o')

        plt.tight_layout()
        plt.show()
        

# Visualize predictions with uncertainty
visualize_predictions_with_uncertainty(model, test_dataset, class_names, uncertainties)