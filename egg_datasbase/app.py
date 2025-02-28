
import streamlit as st
import joblib
import numpy as np
from scipy.signal import butter, filtfilt
from scipy.fft import fft

# Load your model
model_path = '/content/drive/MyDrive/MACHINE LEARNING PROJECT/EGG_dataset/egg_datasbase/models/noisy_0.05_Naive_Bayes.joblib'
model = joblib.load(model_path)

# Bandpass filter for noise reduction
fs = 2  # Hz, sampling frequency
N = 4096  # number of points for FFT analysis
b, a = butter(3, [0.03, 0.25], btype='band', fs=fs)

def bandpass_filter(eggsignal, fs, lowcut, highcut):
    nyquist = fs / 2
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(3, [low, high], 'bandpass')
    return filtfilt(b, a, eggsignal)

def calculate_dominant_frequency(data, fs, N):
    ch1 = bandpass_filter(data[:, 0], fs, 0.03, 0.25)  # Channel 1
    ch2 = bandpass_filter(data[:, 1], fs, 0.03, 0.25)  # Channel 2
    ch3 = bandpass_filter(data[:, 2], fs, 0.03, 0.25)  # Channel 3

    fch1 = np.abs(fft(ch1, N))**2
    fch1 = fch1[:(N//2+1)]
    fch2 = np.abs(fft(ch2, N))**2
    fch2 = fch2[:(N//2+1)]
    fch3 = np.abs(fft(ch3, N))**2
    fch3 = fch3[:(N//2+1)]

    b1 = np.argmax(fch1) / 2048
    b2 = np.argmax(fch2) / 2048
    b3 = np.argmax(fch3) / 2048

    return [b1 * 60, b2 * 60, b3 * 60]  # Convert to cycles per minute (cpm)

def extract_features(eggsignal, fs):
    mean_val = np.mean(eggsignal)
    median_val = np.median(eggsignal)
    std_val = np.std(eggsignal)
    peak_to_peak = np.ptp(eggsignal)
    zero_crossings = np.count_nonzero(np.diff(np.sign(eggsignal)))  # Zero crossings
    spectral_entropy = np.sum(np.abs(fft(eggsignal))**2)  # Total spectral power as an approximation for entropy
    peak_freq = np.argmax(np.abs(fft(eggsignal)))  # Peak frequency in the signal

    # Dominant frequency
    N = len(eggsignal)
    dominant_freq = np.argmax(np.abs(fft(eggsignal))) / fs

    # Total power in the signal
    total_power = np.sum(np.abs(fft(eggsignal))**2)

    return {
        'mean': mean_val,
        'median': median_val,
        'std_dev': std_val,
        'peak_to_peak': peak_to_peak,
        'zero_crossings': zero_crossings,
        'dominant_frequency': dominant_freq,
        'spectral_entropy': spectral_entropy,
        'peak_frequency': peak_freq,
        'total_power': total_power
    }

st.title('Egg Prediction Model')

# Input for fasting state
st.header('Fasting State')
user_input_fasting_ch1 = st.text_area('Enter Channel 1 data (comma-separated values):', key='fasting_ch1')
user_input_fasting_ch2 = st.text_area('Enter Channel 2 data (comma-separated values):', key='fasting_ch2')
user_input_fasting_ch3 = st.text_area('Enter Channel 3 data (comma-separated values):', key='fasting_ch3')

# Input for postprandial state
st.header('Postprandial State')
user_input_postprandial_ch1 = st.text_area('Enter Channel 1 data (comma-separated values):', key='postprandial_ch1')
user_input_postprandial_ch2 = st.text_area('Enter Channel 2 data (comma-separated values):', key='postprandial_ch2')
user_input_postprandial_ch3 = st.text_area('Enter Channel 3 data (comma-separated values):', key='postprandial_ch3')

if st.button('Predict'):
    try:
        # Process input data for fasting state
        signal_fasting_ch1 = np.fromstring(user_input_fasting_ch1, sep=',')
        signal_fasting_ch2 = np.fromstring(user_input_fasting_ch2, sep=',')
        signal_fasting_ch3 = np.fromstring(user_input_fasting_ch3, sep=',')

        features_fasting_ch1 = extract_features(signal_fasting_ch1, fs)
        features_fasting_ch2 = extract_features(signal_fasting_ch2, fs)
        features_fasting_ch3 = extract_features(signal_fasting_ch3, fs)

        df_fasting = calculate_dominant_frequency(
            np.column_stack((signal_fasting_ch1, signal_fasting_ch2, signal_fasting_ch3)), fs, N)

        # Process input data for postprandial state
        signal_postprandial_ch1 = np.fromstring(user_input_postprandial_ch1, sep=',')
        signal_postprandial_ch2 = np.fromstring(user_input_postprandial_ch2, sep=',')
        signal_postprandial_ch3 = np.fromstring(user_input_postprandial_ch3, sep=',')

        features_postprandial_ch1 = extract_features(signal_postprandial_ch1, fs)
        features_postprandial_ch2 = extract_features(signal_postprandial_ch2, fs)
        features_postprandial_ch3 = extract_features(signal_postprandial_ch3, fs)

        df_postprandial = calculate_dominant_frequency(
            np.column_stack((signal_postprandial_ch1, signal_postprandial_ch2, signal_postprandial_ch3)), fs, N)

        # Combine all features for model input
        model_input = [
            features_fasting_ch1['mean'], features_fasting_ch1['median'], features_fasting_ch1['std_dev'],
            features_fasting_ch1['peak_to_peak'], features_fasting_ch1['zero_crossings'], features_fasting_ch1['dominant_frequency'],
            features_fasting_ch1['spectral_entropy'], features_fasting_ch1['peak_frequency'], features_fasting_ch1['total_power'],

            features_fasting_ch2['mean'], features_fasting_ch2['median'], features_fasting_ch2['std_dev'],
            features_fasting_ch2['peak_to_peak'], features_fasting_ch2['zero_crossings'], features_fasting_ch2['dominant_frequency'],
            features_fasting_ch2['spectral_entropy'], features_fasting_ch2['peak_frequency'], features_fasting_ch2['total_power'],

            features_fasting_ch3['mean'], features_fasting_ch3['median'], features_fasting_ch3['std_dev'],
            features_fasting_ch3['peak_to_peak'], features_fasting_ch3['zero_crossings'], features_fasting_ch3['dominant_frequency'],
            features_fasting_ch3['spectral_entropy'], features_fasting_ch3['peak_frequency'], features_fasting_ch3['total_power'],

            df_fasting[0], df_fasting[1], df_fasting[2],

            features_postprandial_ch1['mean'], features_postprandial_ch1['median'], features_postprandial_ch1['std_dev'],
            features_postprandial_ch1['peak_to_peak'], features_postprandial_ch1['zero_crossings'], features_postprandial_ch1['dominant_frequency'],
            features_postprandial_ch1['spectral_entropy'], features_postprandial_ch1['peak_frequency'], features_postprandial_ch1['total_power'],

            features_postprandial_ch2['mean'], features_postprandial_ch2['median'], features_postprandial_ch2['std_dev'],
            features_postprandial_ch2['peak_to_peak'], features_postprandial_ch2['zero_crossings'], features_postprandial_ch2['dominant_frequency'],
            features_postprandial_ch2['spectral_entropy'], features_postprandial_ch2['peak_frequency'], features_postprandial_ch2['total_power'],

            features_postprandial_ch3['mean'], features_postprandial_ch3['median'], features_postprandial_ch3['std_dev'],
            features_postprandial_ch3['peak_to_peak'], features_postprandial_ch3['zero_crossings'], features_postprandial_ch3['dominant_frequency'],
            features_postprandial_ch3['spectral_entropy'], features_postprandial_ch3['peak_frequency'], features_postprandial_ch3['total_power'],

            df_postprandial[0], df_postprandial[1], df_postprandial[2]
        ]

        # Make prediction
        prediction = model.predict([model_input])

        st.write(f'Dominant Frequencies (Fasting): {df_fasting[0]} cpm, {df_fasting[1]} cpm, {df_fasting[2]} cpm')
        st.write(f'Dominant Frequencies (Postprandial): {df_postprandial[0]} cpm, {df_postprandial[1]} cpm, {df_postprandial[2]} cpm')
        st.write(f'Prediction: {prediction[0]}')
    except Exception as e:
        st.write('Error processing input data. Please ensure it is in the correct format.')
        st.write(str(e))
