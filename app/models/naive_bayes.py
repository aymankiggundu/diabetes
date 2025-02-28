import joblib
import numpy as np
from sklearn.naive_bayes import GaussianNB
from ..utils.signal_processing import extract_features

class DiabetesPredictor:
    def __init__(self, model_path):
        self.model = joblib.load(model_path)
        self.fs = 2
    
    def prepare_input(self, fasting_data, postprandial_data):
        features = []
        for data in [fasting_data, postprandial_data]:
            for channel in range(3):
                signal = data.iloc[:, channel].values
                features.extend(extract_features(signal, self.fs))
        return features[:17]  # Take first 17 features
    
    def predict(self, features):
        prediction = self.model.predict([features])
        probability = self.model.predict_proba([features])[0]
        return {
            'prediction': 'Diabetic' if prediction[0] == 1 else 'Non-diabetic',
            'probability': float(max(probability)),
            'features_extracted': len(features)
        }
