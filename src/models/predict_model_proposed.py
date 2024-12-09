import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import seaborn as sns

class_names = ['Mild Demented', 'Moderate Demented', 'Non Demented', 'Very MildDemented']

# --------------------------------------------------------------
# Prediction and Evaluation
# --------------------------------------------------------------
def evaluate_model(model, dataset, class_names):
    """
    Evaluate the model on a dataset and return detailed metrics.
    
    Args:
    - model: Trained TensorFlow model.
    - dataset: tf.data.Dataset to evaluate.
    - class_names: List of class names.
    
    Returns:
    - metrics_df: DataFrame containing per-class metrics.
    - avg_metrics: Dictionary of micro, macro, and weighted averages.
    - true_labels: Ground truth labels.
    - predictions: Predicted labels.
    """
    true_labels = []
    predictions = []
    for images, labels in dataset:
        preds = model.predict(images)
        predictions.extend(np.argmax(preds, axis=1))
        true_labels.extend(np.argmax(labels, axis=1))
    
    # Classification report
    report = classification_report(
        true_labels,
        predictions,
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )
    
    # Create a DataFrame for per-class metrics
    metrics_df = pd.DataFrame(report).transpose()
    class_metrics = metrics_df.iloc[:-3, :]  # Exclude micro/macro/weighted averages
    avg_metrics = metrics_df.iloc[-3:, :]  # Micro, macro, weighted averages
    
    return class_metrics, avg_metrics, true_labels, predictions

def display_metrics(class_metrics, avg_metrics):
    """
    Display evaluation metrics in tabular form.
    """
    print("\nPer-Class Metrics:")
    print(class_metrics.to_string(index=True))
    print("\nAverage Metrics:")
    print(avg_metrics.to_string(index=True))

# Evaluate the model
class_metrics, avg_metrics, true_labels, predictions = evaluate_model(model, test_dataset, class_names)

# Display metrics
display_metrics(class_metrics, avg_metrics)

# --------------------------------------------------------------
# Confusion Matrix
# --------------------------------------------------------------
def plot_confusion_matrix(true_labels, predictions, class_names):
    """
    Plot a confusion matrix.
    """
    cm = confusion_matrix(true_labels, predictions)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title("Confusion Matrix")
    plt.show()

plot_confusion_matrix(true_labels, predictions, class_names)

# --------------------------------------------------------------
# Visualization with Predictions
# --------------------------------------------------------------
def display_predictions_with_colors(model, dataset, class_names, num_images=10):
    """
    Display images with predicted and true labels. Highlight mismatched predictions.
    """
    for images, labels in dataset.take(1):
        preds = model.predict(images)
        preds = np.argmax(preds, axis=1)
        labels = np.argmax(labels, axis=1)
        
        images = (images + 1) / 2.0  # Scale images to [0, 1]
        
        plt.figure(figsize=(15, 10))
        for i in range(num_images):
            plt.subplot(2, 5, i + 1)
            img = images[i].numpy()
            plt.imshow(img)
            
            true_label = class_names[labels[i]]
            pred_label = class_names[preds[i]]
            
            # Set color based on correctness
            color = "green" if true_label == pred_label else "red"
            plt.title(f"True: {true_label}\nPred: {pred_label}", color=color)
            plt.axis('off')
        plt.tight_layout()
        plt.show()

# Display predictions with colors
display_predictions_with_colors(model, test_dataset, class_names)