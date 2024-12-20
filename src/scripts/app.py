from flask import Flask, request, jsonify, render_template
import os
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.applications.efficientnet import preprocess_input
import tempfile
import base64

# Initialize Flask app
app = Flask(
    __name__,
    static_folder="../static",
    template_folder="../",
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

def preprocess_image(image_path):
    img = Image.open(image_path).resize((224, 224)).convert("RGB")
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return preprocess_input(img_array)

def predict_with_uncertainty(model, image, n_samples=10):
    predictions = [model(image, training=True) for _ in range(n_samples)]
    predictions = np.stack(predictions, axis=0)
    mean_preds = np.mean(predictions, axis=0)
    uncertainty = np.std(predictions, axis=0)
    return mean_preds, uncertainty

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    if file:
        try:
            # Create a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as temp_file:
                file.save(temp_file.name)
                filepath = temp_file.name

            img_array = preprocess_image(filepath)
            mean_preds, uncertainty = predict_with_uncertainty(model, img_array, n_samples=10)
            predicted_label = class_labels[np.argmax(mean_preds)]
            confidence = 1 - uncertainty.flatten()[0]
            
            # Read the original image and convert to base64
            with open(filepath, "rb") as image_file:
                encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
            
            # Remove the temporary file
            os.unlink(filepath)
            
            return jsonify({
                "predicted_label": predicted_label, 
                "confidence": float(confidence),
                "image": encoded_image
            })
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    
    return jsonify({"error": "Invalid file"}), 400

if __name__ == "__main__":
    app.run(debug=True)

