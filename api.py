from flask import Flask, request, jsonify
import numpy as np
from keras.models import load_model
import cv2
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={r"/predict": {"origins": "*"}})

# Load the saved model
model = load_model('wood_species_model.h5')

# Mapping class indices to wood type names
class_names = {
    0: 'Bubinga',
    1: 'Cherry',
    2: 'Sycamore',
    3: 'Douglas Fir'
    # Add more mappings as needed
}


@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    file = request.files['image']
    image = cv2.imdecode(np.frombuffer(
        file.read(), np.uint8), cv2.IMREAD_COLOR)
    image = cv2.resize(image, (50, 50))
    image = image.astype('float32') / 255
    image = np.expand_dims(image, axis=0)

    # Log the shape and content of the image
    print(f"Image shape: {image.shape}")

    prediction = model.predict(image)
    class_index = np.argmax(prediction)
    confidence = float(np.max(prediction))

    # Log the prediction
    print(f"Prediction: {prediction}")

    # Update with your actual wood types
    wood_types = ['Bubinga', 'Cherry', 'Sycamore', 'Douglas Fir']
    return jsonify({"class_index": int(class_index), "confidence": confidence, "wood_type": wood_types[class_index]})


if __name__ == '__main__':
    app.run(debug=True)
