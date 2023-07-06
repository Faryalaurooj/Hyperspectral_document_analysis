# author Faryal Aurooj 
# This code demonstrates how to extract spectral information
# for different inks from a hyperspectral image, visualize the spectral
# responses, and analyze the reflectance characteristics at 
# different wavelengths in a single 2D plot

import cv2
import matplotlib.pyplot as plt
import numpy as np
import spectral

# Load the hyperspectral image
img = spectral.open_image('input/w01_p02_corrected.hdr').load()

# Load the grayscale image
crop_img = cv2.imread('output/hyperspectral_image_grey_scale.png', cv2.IMREAD_GRAYSCALE)

# Plot the cropped image and the histogram of all pixels
plt.figure(figsize=(15,10))
plt.subplot(221)
plt.imshow(crop_img, cmap='gray')
plt.title('Cropped Image')
plt.xlim([180, 475])  # set x-axis limits
plt.ylim([420, 55])   # set y-axis limits
plt.subplot(222)
plt.hist(crop_img.ravel())
plt.title('Pixel Intensity Distribution in the Grayscale Image')  # Updated title
plt.xlabel('Pixel Intensity')  # X-axis label
plt.ylabel('Frequency')  # Y-axis label

plt.tight_layout()  # Adjust the layout
plt.show()

if img is not None:
    # Extract the cube for the cropped image
    crop_cube = img[60:630, 40:462, :149]

    # Apply thresholding to segment the image and extract the foreground pixels
    cropped_img = crop_cube[60:630, 40:462, 29]
    cropped_cube = crop_cube[60:630, 40:462, :149]
    bi_image = cv2.threshold(cropped_img, 0.8, 1.05, cv2.THRESH_BINARY)[1]
    line_ranges = [(0, 37), (45, 85), (95, 135), (142, 182), (190, 230)]
    n_lines = len(line_ranges)

    ink_masks = [bi_image[r[0]:r[1], 0:422] < 0.2 for r in line_ranges]
    line_cubes = [cropped_cube[r[0]:r[1], 0:422, :149] for r in line_ranges]

    # Get the spectral responses of foreground pixels for each ink
    ink_spectra = []
    for cube, mask in zip(line_cubes, ink_masks):
        ink_cube = cube * mask[..., None]
        ink_pixels = ink_cube[ink_cube != 0].reshape(-1, cube.shape[-1])
        ink_spectra.append(ink_pixels)

    # Get the wavelengths
    wavelengths = img.metadata['wavelength']

    # Plot the spectral responses of foreground pixels for each ink
    plt.figure(figsize=(15, 10))
    colors = plt.cm.rainbow(np.linspace(0, 1, len(ink_spectra)))  # Generate a color map

    for i, spectra in enumerate(ink_spectra):
        plt.plot(wavelengths, np.mean(spectra, axis=0), color=colors[i])

    plt.title("Spectral Responses of Inks")
    plt.xlabel("Wavelength")
    plt.ylabel("Reflectance")
    plt.legend([f"Ink {i+1}" for i in range(len(ink_spectra))])  # Add a legend with ink labels

 # Set the x-axis ticks to show specific mean wavelength values
   # mean_wavelengths = [400, 500,
    plt.tight_layout()  # Adjust the layout to prevent overlapping
    plt.show()
else:
    print("Error loading image")

