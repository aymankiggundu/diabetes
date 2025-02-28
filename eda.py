import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
file_path = "dataset.csv"
data = pd.read_csv(file_path)

# Display the first few rows
print("Dataset Preview:")§
print(data.head())

# Check for missing values
print("\nMissing Values:")
print(data.isnull().sum())

# Summary statistics
print("\nSummary Statistics:")
print(data.describe())

# Visualizations
plt.figure(figsize=(12, 6))
sns.histplot(data['Glucose_Level'], bins=30, kde=True, color='blue')
plt.title("Distribution of Glucose Levels")
plt.xlabel("Glucose Level (mg/dL)")
plt.ylabel("Count")
plt.show()

plt.figure(figsize=(12, 6))
sns.histplot(data['EGG_Frequency'], bins=30, kde=True, color='green')
plt.title("Distribution of EGG Frequency")
plt.xlabel("EGG Frequency (Hz)")
plt.ylabel("Count")
plt.show()

# Diabetes prevalence
plt.figure(figsize=(6, 4))
sns.countplot(x='Diabetes_Status', data=data, palette=['red', 'green'])
plt.title("Diabetes Prevalence in Dataset")
plt.xlabel("Diabetes Status (0 = No, 1 = Yes)")
plt.ylabel("Count")
plt.show()

# Correlation heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title("Feature Correlation Heatmap")
plt.show()


