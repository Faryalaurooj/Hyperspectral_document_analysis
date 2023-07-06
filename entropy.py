# This code calculates the entropy of each channel, 
# which represents the quantity of information in numerical form. 
# The entropy is calculated based on the 
# probability distribution of pixel intensities in each channel.

import cv2
import numpy as np

def calculate_entropy(image):
    # Normalize pixel values to [0, 1]
    normalized_image = image.astype(float) / 255.0

    # Calculate histogram
    histogram = np.histogram(normalized_image, bins=256, range=(0, 1))[0]

    # Calculate probability distribution
    probabilities = histogram / np.sum(histogram)

    # Calculate entropy
    entropy = -np.sum(probabilities * np.log2(probabilities + 1e-7))

    return entropy

# Load the red, green, and blue channel images
red_channel = cv2.imread('output/red channel.png', 0)  # Read as grayscale
green_channel = cv2.imread('output/green channel.png', 0)
blue_channel = cv2.imread('output/blue channel.png', 0)

# Calculate the entropy for each channel
red_entropy = calculate_entropy(red_channel)
green_entropy = calculate_entropy(green_channel)
blue_entropy = calculate_entropy(blue_channel)

# Print the entropy values
print("Red Channel Entropy:", red_entropy)
print("Green Channel Entropy:", green_entropy)
print("Blue Channel Entropy:", blue_entropy)
