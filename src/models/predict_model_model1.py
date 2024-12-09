import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import pandas as pd

# --------------------------------------------------------------
# Prediction and Evaluation
# --------------------------------------------------------------
def evaluate_model(model, generator, class_names):
    """
    Evaluate the model on a generator and return detailed metrics.

    Args:
    - model: Trained TensorFlow model.
    - generator: ImageDataGenerator to evaluate.
    - class_names: List of class names.

    Returns:
    - class_metrics: DataFrame containing per-class metrics.
    - avg_metrics: Dictionary of micro, macro, and weighted averages.
    - true_labels: Ground truth labels.
    - predictions: Predicted labels.
    """
    predictions = model.predict(generator)
    pred_labels = np.argmax(predictions, axis=1)
    true_labels = generator.classes  # Ground truth labels
    class_names = list(generator.class_indices.keys())

    # Generate classification report
    report = classification_report(
        true_labels,
        pred_labels,
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )
    
    # Create DataFrame for per-class metrics
    metrics_df = pd.DataFrame(report).transpose()
    class_metrics = metrics_df.iloc[:-3, :]  # Exclude micro/macro/weighted averages
    avg_metrics = metrics_df.iloc[-3:, :]  # Micro, macro, weighted averages

    return class_metrics, avg_metrics, true_labels, pred_labels

def display_metrics(class_metrics, avg_metrics):
    """
    Display evaluation metrics in tabular form.
    """
    print("\nPer-Class Metrics:")
    print(class_metrics.to_string(index=True))
    print("\nAverage Metrics:")
    print(avg_metrics.to_string(index=True))

# Evaluate the model
class_metrics, avg_metrics, true_labels, pred_labels = evaluate_model(model1, test, classes)

# Display metrics
display_metrics(class_metrics, avg_metrics)

# --------------------------------------------------------------
# Confusion Matrix
# --------------------------------------------------------------
def plot_confusion_matrix(true_labels, pred_labels, class_names):
    """
    Plot a confusion matrix.
    """
    cm = confusion_matrix(true_labels, pred_labels)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title("Confusion Matrix")
    plt.show()

plot_confusion_matrix(true_labels, pred_labels, classes)

# --------------------------------------------------------------
# Visualization with Predictions
# --------------------------------------------------------------
def display_predictions_with_colors(model, generator, class_names, num_images=10):
    """
    Display images with predicted and true labels. Highlight mismatched predictions.
    """
    images, labels = next(generator)
    predictions = model.predict(images)
    pred_labels = np.argmax(predictions, axis=1)
    true_labels = np.argmax(labels, axis=1)

    # Rescale images for display
    images = (images + 1) / 2.0

    plt.figure(figsize=(15, 10))
    for i in range(min(num_images, len(images))):
        plt.subplot(2, 5, i + 1)
        plt.imshow(images[i])
        true_label = class_names[true_labels[i]]
        pred_label = class_names[pred_labels[i]]
        
        # Set color based on prediction correctness
        color = "green" if true_label == pred_label else "red"
        plt.title(f"True: {true_label}\nPred: {pred_label}", color=color)
        plt.axis('off')
    plt.tight_layout()
    plt.show()

# Display predictions with true/false colors
display_predictions_with_colors(model1, test, classes, num_images=10)
