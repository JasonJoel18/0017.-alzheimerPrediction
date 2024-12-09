import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os

model1 = load_model('/Volumes/Jason\'s T7/2. Education/Research/Thesis/Paper/0017. alzheimerPrediction/models/model1.h5')
model2 = load_model('/Volumes/Jason\'s T7/2. Education/Research/Thesis/Paper/0017. alzheimerPrediction/models/Osteoporosis_Model_binary.h5')


from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, precision_score, recall_score

# Evaluate the model
test_accuracy = accuracy_score(actual_labels, predicted_labels)
print(f"Test Accuracy: {test_accuracy:.4f}")

# Generate a classification report with all average types
report = classification_report(actual_labels, predicted_labels, target_names=class_names, digits=4)
print("\nClassification Report:\n", report)

# Calculate and print specific metrics with macro and weighted averages
macro_f1 = f1_score(actual_labels, predicted_labels, average='macro')
weighted_f1 = f1_score(actual_labels, predicted_labels, average='weighted')

macro_precision = precision_score(actual_labels, predicted_labels, average='macro')
weighted_precision = precision_score(actual_labels, predicted_labels, average='weighted')

macro_recall = recall_score(actual_labels, predicted_labels, average='macro')
weighted_recall = recall_score(actual_labels, predicted_labels, average='weighted')

print(f"Macro F1-Score: {macro_f1:.4f}")
print(f"Weighted F1-Score: {weighted_f1:.4f}")
print(f"Macro Precision: {macro_precision:.4f}")
print(f"Weighted Precision: {weighted_precision:.4f}")
print(f"Macro Recall: {macro_recall:.4f}")
print(f"Weighted Recall: {weighted_recall:.4f}")

# Plot the confusion matrix
cm = confusion_matrix(actual_labels, predicted_labels)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()