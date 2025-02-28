from flask import render_template, request, jsonify, send_file
import pandas as pd
import numpy as np
import io
from . import app
from .models.naive_bayes import DiabetesPredictor

predictor = DiabetesPredictor('./models/saved/noisy_0.05_Naive_Bayes.joblib')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        fasting_file = request.files['fasting_file']
        postprandial_file = request.files['postprandial_file']

        fasting_data = pd.read_csv(fasting_file)
        postprandial_data = pd.read_csv(postprandial_file)

        features = predictor.prepare_input(fasting_data, postprandial_data)
        result = predictor.predict(features)

        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/sample_data')
def get_sample_data():
    sample_data = pd.DataFrame({
        'Channel1': np.linspace(0.5, 2.5, 21),
        'Channel2': np.linspace(0.6, 2.6, 21),
        'Channel3': np.linspace(0.7, 2.7, 21)
    })
    
    buffer = io.StringIO()
    sample_data.to_csv(buffer, index=False)
    buffer.seek(0)
    
    return send_file(
        io.BytesIO(buffer.getvalue().encode()),
        mimetype='text/csv',
        as_attachment=True,
        download_name='sample_data.csv'
    )
