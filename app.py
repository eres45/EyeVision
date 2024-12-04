from flask import Flask, request, render_template, jsonify
from keras.models import load_model
import numpy as np
import cv2
from PIL import Image, ImageEnhance
import traceback

app = Flask(__name__)

# Load the Keras model
model = load_model('model_after_testing.keras')

def detect_blur(image):
    """Detect if an image is blurry using Laplacian variance."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    return variance < 100  # Threshold for blur detection (tune this based on testing)

def preprocess_image(file_stream):
    """Preprocess the image for the model: crop, resize, enhance, and normalize."""
    try:
        # Load image with OpenCV
        file_bytes = np.frombuffer(file_stream.read(), np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        # Detect blur
        if detect_blur(image):
            raise ValueError("The uploaded image is too blurry. Please upload a clearer image.")

        # Detect the eye region using Haar Cascade (or pre-trained detector)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        if len(eyes) == 0:
            raise ValueError("No eyes detected in the image. Please upload a valid eye image.")

        # Crop the first detected eye
        x, y, w, h = eyes[0]
        cropped_eye = image[y:y+h, x:x+w]

        # Resize to model's input size
        resized_eye = cv2.resize(cropped_eye, (224, 224))

        # Enhance contrast and brightness
        enhanced_eye = cv2.convertScaleAbs(resized_eye, alpha=1.3, beta=30)

        # Normalize pixel values
        normalized_eye = enhanced_eye / 255.0
        normalized_eye = np.expand_dims(normalized_eye, axis=0)  # Add batch dimension

        return normalized_eye
    except Exception as e:
        raise ValueError(f"Image preprocessing failed: {str(e)}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        file = request.files.get('file')
        if not file:
            return jsonify({'error': 'No file uploaded'}), 400

        # Preprocess the image
        processed_image = preprocess_image(file)

        # Perform prediction
        prediction = model.predict(processed_image)[0][0]
        print(f"Raw prediction value: {prediction}")  # Debug log

        # Determine result
        result = "Eye Flu Detected" if prediction >= 0.5 else "Healthy Eye"

        # Return response
        return jsonify({'result': result, 'confidence': float(prediction)})

    except Exception as e:
        traceback.print_exc()  # Debugging stack trace
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
