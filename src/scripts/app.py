from flask import Flask, request, jsonify, render_template
import os
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.applications.efficientnet import preprocess_input
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Initialize Flask app
app = Flask(
    __name__,
    static_folder="../static",  # Locate `static/` at `src/static`
    template_folder="../",  # Locate `index.html` at `src/`
)

# Load your trained model
model = tf.keras.models.load_model(
    "/Volumes/JasonT7/2.Education/Research/Thesis/Paper/0017. alzheimerPrediction/src/scripts/jason_alzheimer_prediction_model.keras"
)

# Define class labels
class_labels = [
    "Mild Demented",
    "Moderate Demented",
    "Non Demented",
    "Very Mild Demented",
]


# Preprocess image
def preprocess_image(image_path):
    img = Image.open(image_path).resize((224, 224)).convert("RGB")
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return preprocess_input(img_array)


# Predict with uncertainty
def predict_with_uncertainty(model, image, n_samples=10):
    predictions = [model(image, training=True) for _ in range(n_samples)]
    predictions = tf.stack(predictions, axis=0)
    mean_preds = tf.reduce_mean(predictions, axis=0).numpy()
    uncertainty = tf.math.reduce_std(predictions, axis=0).numpy()
    return mean_preds, uncertainty


# Visualization
def visualize_prediction_with_uncertainty(image_path, predicted_label, confidence):
    img = Image.open(image_path).resize((224, 224))
    colors = ["#FF4C4C", "#FFD34F", "#4CFF4C"]
    cmap = mcolors.LinearSegmentedColormap.from_list("custom_gradient", colors, N=100)

    plt.figure(figsize=(8, 6))
    plt.imshow(img)
    plt.axis("off")
    plt.title(
        f"Predicted: {predicted_label}\nConfidence: {confidence * 100:.2f}%",
        fontsize=14,
        color="green" if confidence > 0.85 else "red",
    )
    output_path = os.path.join("src/static", "output.png")
    plt.savefig(output_path)
    plt.close()


# Flask routes
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["file"]
    if file:
        filepath = os.path.join("../static/uploads", file.filename)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        file.save(filepath)
        img_array = preprocess_image(filepath)
        mean_preds, uncertainty = predict_with_uncertainty(
            model, img_array, n_samples=10
        )
        predicted_label = class_labels[np.argmax(mean_preds)]
        confidence = 1 - uncertainty.flatten()[0]
        visualize_prediction_with_uncertainty(filepath, predicted_label, confidence)
        return jsonify({"predicted_label": predicted_label, "confidence": confidence})
    return jsonify({"error": "No file uploaded"}), 400


if __name__ == "__main__":
    app.run(debug=True)
