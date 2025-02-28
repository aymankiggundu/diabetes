from flask import Flask, render_template, request, jsonify, send_file
import numpy as np
import pandas as pd
import joblib
from scipy.signal import butter, filtfilt
from scipy.fft import fft
import io
import os

app = Flask(__name__)

# Load the model
model_path = './noisy_0.05_Naive_Bayes.joblib'
model = joblib.load(model_path)

# Signal processing functions
def bandpass_filter(eggsignal, fs, lowcut, highcut):
    nyquist = fs / 2
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(3, [low, high], 'bandpass')
    return filtfilt(b, a, eggsignal)

def extract_features(eggsignal, fs=2):
    mean_val = np.mean(eggsignal)
    std_val = np.std(eggsignal)
    fft_vals = np.abs(fft(eggsignal))
    dominant_freq = np.argmax(fft_vals) / fs
    return [mean_val, std_val, dominant_freq]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get files from request
        fasting_file = request.files['fasting_file']
        postprandial_file = request.files['postprandial_file']

        # Read CSV files
        fasting_data = pd.read_csv(fasting_file)
        postprandial_data = pd.read_csv(postprandial_file)

        # Extract features
        features = []
        for data in [fasting_data, postprandial_data]:
            for channel in range(3):
                signal = data.iloc[:, channel].values
                features.extend(extract_features(signal))

        # Take first 17 features
        model_input = features[:17]

        # Make prediction
        prediction = model.predict([model_input])
        probability = model.predict_proba([model_input])[0]

        result = {
            'prediction': 'Diabetic' if prediction[0] == 1 else 'Non-diabetic',
            'probability': float(max(probability)),
            'features_extracted': len(model_input)
        }

        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/sample_data')
def get_sample_data():
    # Create sample data
    sample_data = pd.DataFrame({
        'Channel1': np.linspace(0.5, 2.5, 21),
        'Channel2': np.linspace(0.6, 2.6, 21),
        'Channel3': np.linspace(0.7, 2.7, 21)
    })
    
    # Create buffer
    buffer = io.StringIO()
    sample_data.to_csv(buffer, index=False)
    buffer.seek(0)
    
    return send_file(
        io.BytesIO(buffer.getvalue().encode()),
        mimetype='text/csv',
        as_attachment=True,
        download_name='sample_data.csv'
    )

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0")
