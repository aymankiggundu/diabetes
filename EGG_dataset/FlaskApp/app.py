
from flask import Flask, request, jsonify
import joblib
import numpy as np

# Initialize the Flask app
app = Flask(__name__)

# Load your saved model (make sure it's in the right directory on Google Drive)
model = joblib.load('/content/drive/MyDrive/FlaskApp/naive_bayes_model.pkl')

# Define the routes
@app.route('/')
def home():
    return "Welcome to the ML model API!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()  # Get the input data
        features = np.array(data['features']).reshape(1, -1)  # Reshape for prediction
        prediction = model.predict(features)
        return jsonify({'prediction': prediction[0]})
    except Exception as e:
        return jsonify({'error': str(e)})

# Run the app if this file is run directly
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
