from flask import Flask, request, jsonify, render_template
import os
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.applications.efficientnet import preprocess_input
import tempfile
import base64

app = Flask(
    __name__,
    static_folder="../static",
    template_folder="../",
)

model = tf.keras.models.load_model(
    "/Volumes/JasonT7/2.Education/Research/Thesis/Paper/0017. alzheimerPrediction/src/scripts/jason_alzheimer_prediction_model_23788_images_10_epochs.keras"
)

class_labels = [
    "Mild Demented",
    "Moderate Demented",
    "Non Demented",
    "Very Mild Demented",
]


def preprocess_image(image_path):
    img = Image.open(image_path).resize((224, 224)).convert("RGB")
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return preprocess_input(img_array)


def predict_with_uncertainty(model, image, n_samples=50):
    predictions = []
    for _ in range(n_samples):
        pred = model(image, training=True)
        predictions.append(pred.numpy())
    predictions = np.stack(predictions, axis=0)
    mean_preds = np.mean(predictions, axis=0)
    uncertainty = np.std(predictions, axis=0)
    predicted_class = np.argmax(mean_preds)
    class_uncertainty = uncertainty[0][predicted_class]
    confidence = 1 - class_uncertainty

    return mean_preds[0], confidence, predicted_class


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    if file:
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_file:
                file.save(temp_file.name)
                filepath = temp_file.name
            img_array = preprocess_image(filepath)
            predictions, confidence, predicted_class = predict_with_uncertainty(
                model, img_array, n_samples=10
            )
            predicted_label = class_labels[predicted_class]
            with open(filepath, "rb") as image_file:
                encoded_image = base64.b64encode(image_file.read()).decode("utf-8")
            os.unlink(filepath)
            return jsonify(
                {
                    "predicted_label": predicted_label,
                    "confidence": float(confidence),
                    "predictions": predictions.tolist(),
                    "image": encoded_image,
                }
            )
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    return jsonify({"error": "Invalid file"}), 400


if __name__ == "__main__":
    app.run(debug=True)
