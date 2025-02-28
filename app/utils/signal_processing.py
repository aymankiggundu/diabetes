import numpy as np
from scipy.signal import butter, filtfilt
from scipy.fft import fft

def bandpass_filter(eggsignal, fs, lowcut, highcut):
    """Apply bandpass filter to EGG signal."""
    nyquist = fs / 2
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(3, [low, high], 'bandpass')
    return filtfilt(b, a, eggsignal)

def extract_features(eggsignal, fs=2):
    """Extract features from EGG signal."""
    mean_val = np.mean(eggsignal)
    std_val = np.std(eggsignal)
    fft_vals = np.abs(fft(eggsignal))
    dominant_freq = np.argmax(fft_vals) / fs
    return [mean_val, std_val, dominant_freq]

def calculate_dominant_frequency(data, fs, N):
    """Calculate dominant frequency for each channel."""
    channels = []
    for i in range(data.shape[1]):
        channel = bandpass_filter(data[:, i], fs, 0.03, 0.25)
        fft_vals = np.abs(fft(channel, N))**2
        fft_vals = fft_vals[:(N//2+1)]
        freq = np.argmax(fft_vals) / 2048 * 60
        channels.append(freq)
    return channels
