import numpy as np
import cv2
from matplotlib import pyplot as plt
import urllib.request
from google.colab import drive

# Mount Google Drive
drive.mount('/content/drive')

# Define the image URL
image_url = "https://media.geeksforgeeks.org/wp-content/uploads/20210903180325/Screenshot1864-238x300.png"

# Download the image
image_path = "/content/cameraman.png"
urllib.request.urlretrieve(image_url, image_path)

# Load the image in grayscale
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Check if the image was loaded successfully
if image is None:
    raise ValueError("Error: Image not found or could not be loaded!")

# Add Gaussian noise to the image
noise = 25 * np.random.randn(*image.shape)  # Noise with a standard deviation of 25
noisy_image = np.clip(image + noise, 0, 255).astype(np.uint8)  # Clip values to valid range

# Apply Gaussian filter with kernel size 5x5 and standard deviation of 1
denoised_5x5 = cv2.GaussianBlur(noisy_image, (5, 5), sigmaX=1)

# Apply Gaussian filter with kernel size 9x9 and standard deviation of 1
denoised_9x9 = cv2.GaussianBlur(noisy_image, (9, 9), sigmaX=1)

# Display the results
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.imshow(noisy_image, cmap='gray')
plt.title('Noisy Image')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(denoised_5x5, cmap='gray')
plt.title('Denoised with 5x5 Gaussian Filter')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(denoised_9x9, cmap='gray')
plt.title('Denoised with 9x9 Gaussian Filter')
plt.axis('off')

plt.tight_layout()
plt.show()
