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

# ROC-AUC Score
y_true = test.classes
y_probs = model.predict(test)
roc_auc = roc_auc_score(y_true, y_probs, multi_class="ovr")
print(f"ROC-AUC Score: {roc_auc:.2f}")

# Confusion Matrix
preds = np.argmax(y_probs, axis=1)
cm = confusion_matrix(y_true, preds)

# Plot Confusion Matrix
plt.figure(figsize=(10, 5))
sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# Grad-CAM for explainability
# Generate Grad-CAM heatmap function
def generate_gradcam(model, image, class_idx, layer_name):
    # Ensure the layer_name is correct
    layer_output = model.get_layer(layer_name).output
    
    # Define Grad-CAM model
    grad_model = Model(inputs=model.input, outputs=[layer_output, model.output])
    
    # Compute gradients
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(image)
        loss = predictions[:, class_idx]
    grads = tape.gradient(loss, conv_outputs)
    
    # Pool gradients over spatial dimensions
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    # Weight the convolution outputs
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(pooled_grads * conv_outputs, axis=-1)
    
    # Normalize the heatmap
    heatmap = np.maximum(heatmap, 0) / np.max(heatmap)
    return heatmap.numpy()

# Test Dataset
sample_image, sample_label = next(iter(test))
class_idx = np.argmax(sample_label[0])

# Fix: Call the model to initialize it
_ = model.predict(tf.expand_dims(sample_image[0], axis=0))

# Check layer names to ensure layer_name exists
print("Model Layers:")
for layer in model.layers:
    print(layer.name)

# Specify the correct layer name
layer_name = "last_conv_layer"  # Replace with the correct name from your model

# Generate Grad-CAM
heatmap = generate_gradcam(model, tf.expand_dims(sample_image[0], axis=0), class_idx, layer_name)

# Visualize Grad-CAM
plt.imshow(sample_image[0])
plt.imshow(heatmap, cmap="jet", alpha=0.5)
plt.title(f"Grad-CAM for Class {class_labels[class_idx]}")
plt.show()

# Monte Carlo Dropout for uncertainty estimation
n_samples = 100
predictions = np.array([model.predict(test) for _ in range(n_samples)])
mean_prediction = predictions.mean(axis=0)
uncertainty = predictions.var(axis=0)

print("Mean Prediction: ", mean_prediction)
print("Uncertainty: ", uncertainty)