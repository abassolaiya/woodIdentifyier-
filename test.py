from flask import Flask, request, jsonify
import numpy as np
from keras.models import load_model
import cv2

app = Flask(__name__)

# Load the model with the correct version
model = load_model('wood_species_model.keras')

# Save the model in a different format (e.g., HDF5)
model.save('wood_species_model.h5')


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

    prediction = model.predict(image)
    class_index = np.argmax(prediction)

    return jsonify({"class_index": int(class_index), "confidence": float(np.max(prediction))})


if __name__ == '__main__':
    app.run(debug=True)
