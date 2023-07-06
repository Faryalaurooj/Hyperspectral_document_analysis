import spectral
import numpy as np
import matplotlib.pyplot as plt

# Load the hyperspectral image
img = spectral.open_image('input/w01_p02_corrected.hdr')
data = img.load()

# Convert the data to grayscale
gray_data = np.mean(data, axis=2)

# Set the brightness and contrast
vmin = np.percentile(gray_data, 5)
vmax = np.percentile(gray_data, 95)

# Display the image with adjusted brightness and contrast
plt.imshow(gray_data, cmap='gray', vmin=vmin, vmax=vmax)
plt.show()

# Print image shape
print(img.shape)

# Get the number of bands
num_bands = img.bands

# Print image shape
print(img.shape)

# Get the starting and ending wavelength range
wavelengths = img.metadata['wavelength']
start_wavelength = wavelengths[0]
end_wavelength = wavelengths[-1]

# Print the results
print("Number of bands: ", num_bands)
print("Starting wavelength: ", start_wavelength)
print("Ending wavelength: ", end_wavelength)