import os

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'your-secret-key-here'
    MODEL_PATH = os.path.join('models', 'saved', 'noisy_0.05_Naive_Bayes.joblib')
    UPLOAD_FOLDER = os.path.join('data', 'uploads')
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size
