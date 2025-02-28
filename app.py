import streamlit as st
import joblib
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
from scipy.fft import fft

# Base path for Google Colab
BASE_PATH = '.'

# Load your model
model_path = f'{BASE_PATH}/noisy_0.05_Naive_Bayes.joblib'
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
    """Extract exactly the required features to match the model's expectations"""
    # Calculate only the essential features to match the model's input size
    mean_val = np.mean(eggsignal)
    std_val = np.std(eggsignal)
    fft_vals = np.abs(fft(eggsignal))
    dominant_freq = np.argmax(fft_vals) / fs
    
    return [mean_val, std_val, dominant_freq]

st.title('EGG Prediction Model')

# File upload section
st.header('Upload Data Files')

# Upload files for fasting and postprandial states
fasting_file = st.file_uploader("Upload Fasting State CSV", type=['csv'], help="CSV file containing fasting state data for all 3 channels")
postprandial_file = st.file_uploader("Upload Postprandial State CSV", type=['csv'], help="CSV file containing postprandial state data for all 3 channels")

if st.button('Predict') and fasting_file is not None and postprandial_file is not None:
    try:
        # Read CSV files
        fasting_data = pd.read_csv(fasting_file)
        postprandial_data = pd.read_csv(postprandial_file)

        # Extract channels from fasting data
        signal_fasting_ch1 = fasting_data.iloc[:, 0].values
        signal_fasting_ch2 = fasting_data.iloc[:, 1].values
        signal_fasting_ch3 = fasting_data.iloc[:, 2].values

        # Extract channels from postprandial data
        signal_postprandial_ch1 = postprandial_data.iloc[:, 0].values
        signal_postprandial_ch2 = postprandial_data.iloc[:, 1].values
        signal_postprandial_ch3 = postprandial_data.iloc[:, 2].values

        # Extract features
        signals = [
            signal_fasting_ch1, signal_fasting_ch2, signal_fasting_ch3,
            signal_postprandial_ch1, signal_postprandial_ch2, signal_postprandial_ch3
        ]

        # Extract features
        features = []
        for signal in signals:
            features.extend(extract_features(signal, fs))
        
        # Take only the first 17 features
        model_input = features[:17]

        # Debug information
        st.write(f"Number of features extracted: {len(model_input)}")
        
        # Make prediction
        prediction = model.predict([model_input])
        
        # Show results with more detailed information
        st.success(f'Prediction: {"Diabetic" if prediction[0] == 1 else "Non-diabetic"}')
        
        # Display data preview
        st.subheader("Data Preview")
        col1, col2 = st.columns(2)
        with col1:
            st.write("Fasting State Data:")
            st.dataframe(fasting_data.head())
        with col2:
            st.write("Postprandial State Data:")
            st.dataframe(postprandial_data.head())
        
    except Exception as e:
        st.error('Error processing input data. Please ensure the CSV files are in the correct format.')
        st.error(str(e))

# Instructions
st.markdown("""
### Instructions:
1. Upload two CSV files:
   - One for fasting state (3 columns for channels 1, 2, and 3)
   - One for postprandial state (3 columns for channels 1, 2, and 3)
2. Each file should have exactly 3 columns representing the three channels
3. The data should be numeric values
4. Click 'Predict' to get the prediction result

### Example CSV Format:
```
Channel1,Channel2,Channel3
0.5,0.6,0.7
0.8,0.9,1.0
...
```
""")

# Add sample data download option
@st.cache_data
def get_sample_data():
    # Create sample data
    sample_data = pd.DataFrame({
        'Channel1': np.linspace(0.5, 2.5, 21),
        'Channel2': np.linspace(0.6, 2.6, 21),
        'Channel3': np.linspace(0.7, 2.7, 21)
    })
    return sample_data

if st.button('Download Sample CSV'):
    sample_data = get_sample_data()
    st.download_button(
        label="Download sample data CSV",
        data=sample_data.to_csv(index=False),
        file_name="sample_data.csv",
        mime="text/csv"
    )