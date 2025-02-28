import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
from scipy.stats import skew, kurtosis
from scipy.fft import fft
import os

# Function for bandpass filter
def bandpass_filter(eggsignal, fs, lowcut, highcut):
    nyquist = fs / 2
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(3, [low, high], 'bandpass')
    return filtfilt(b, a, eggsignal)

# Function to calculate dominant frequency
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

    return [b1, b2, b3]

# Function to extract additional features
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

    # Return the features as a dictionary
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

# Function to label diabetic vs non-diabetic based on feature thresholds
def label_diabetes(features):
    # Diabetes detection thresholds for features (example thresholds)
    diabetes_thresholds = {
        'mean': 0.5,  # Example threshold for 'mean'
        'spectral_entropy': 0.7,  # Example threshold for 'spectral_entropy'
        'dominant_frequency': 0.15,  # Example threshold for 'dominant_frequency'
    }

    # Check if feature values exceed certain thresholds indicating possible diabetes
    if (features['mean'] > diabetes_thresholds['mean'] and
        features['spectral_entropy'] > diabetes_thresholds['spectral_entropy'] and
        features['dominant_frequency'] < diabetes_thresholds['dominant_frequency']):
        return 1  # Diabetic
    else:
        return 0  # Non-diabetic

# File paths
path = '/content/drive/MyDrive/MACHINE LEARNING PROJECT/EGG_dataset/egg_datasbase/'

# Sampling frequency and number of points for FFT
fs = 2  # Hz
N = 4096  # Points for FFT

# List for storing the features
features = []

# Loop through the data for 20 subjects
for ind in range(1, 21):
    # Load fasting data
    file_name_fasting = os.path.join(path, f'ID{ind}_fasting.txt')
    data_fasting = np.loadtxt(file_name_fasting)

    # Extract features for fasting data
    features_fasting_ch1 = extract_features(data_fasting[:, 0], fs)  # Channel 1
    features_fasting_ch2 = extract_features(data_fasting[:, 1], fs)  # Channel 2
    features_fasting_ch3 = extract_features(data_fasting[:, 2], fs)  # Channel 3

    # Calculate dominant frequencies for fasting
    df_fasting = calculate_dominant_frequency(data_fasting, fs, N)

    # Combine features for fasting
    features_fasting = {**features_fasting_ch1, **features_fasting_ch2, **features_fasting_ch3,
                        'df_ch1_fasting': df_fasting[0], 'df_ch2_fasting': df_fasting[1], 'df_ch3_fasting': df_fasting[2],
                        'label': 0}  # 0 for fasting label

    # Label fasting as diabetic or non-diabetic
    features_fasting['diabetes_label'] = label_diabetes(features_fasting)

    # Load postprandial data
    file_name_postprandial = os.path.join(path, f'ID{ind}_postprandial.txt')
    data_postprandial = np.loadtxt(file_name_postprandial)

    # Extract features for postprandial data
    features_postprandial_ch1 = extract_features(data_postprandial[:, 0], fs)  # Channel 1
    features_postprandial_ch2 = extract_features(data_postprandial[:, 1], fs)  # Channel 2
    features_postprandial_ch3 = extract_features(data_postprandial[:, 2], fs)  # Channel 3

    # Calculate dominant frequencies for postprandial
    df_postprandial = calculate_dominant_frequency(data_postprandial, fs, N)

    # Combine features for postprandial
    features_postprandial = {**features_postprandial_ch1, **features_postprandial_ch2, **features_postprandial_ch3,
                             'df_ch1_postprandial': df_postprandial[0], 'df_ch2_postprandial': df_postprandial[1], 'df_ch3_postprandial': df_postprandial[2],
                             'label': 1}  # 1 for postprandial label

    # Label postprandial as diabetic or non-diabetic
    features_postprandial['diabetes_label'] = label_diabetes(features_postprandial)

    # Add both fasting and postprandial features to the list
    features.append(features_fasting)
    features.append(features_postprandial)

# Convert the features list to a DataFrame
df_features = pd.DataFrame(features)

# Saving the features to a CSV file
df_features.to_csv('extracted_features_with_labels.csv', index=False)

# Display a preview of the data
print(df_features.head())
