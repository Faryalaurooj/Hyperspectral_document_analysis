# author: Faryal Aurooj 5/10/2023
#Code applies k-means clustering on the 
#spectral response (wavelengths vs reflectance plot) 
# of ink pixels extracted from a hyperspectral image.
#The number of ink clusters is estimated based on the 
#spectral responses using the elbow method.

import cv2
import numpy as np
import matplotlib.pyplot as plt
import spectral
from sklearn.cluster import KMeans

# Load the hyperspectral image
img = spectral.open_image('input/w01_p02_corrected.hdr').load()

# Load the grayscale image
crop_img = cv2.imread('output/hyperspectral_image_grey_scale.png', cv2.IMREAD_GRAYSCALE)

# now use Spectral Analysis
# Extract the cube corresponding to the cropped image 
# from the hyperspectral image to work with the spectral information
crop_cube = img[60:630, 40:462, :149]

# Apply thresholding to segment the image and extract the foreground pixels
# This is done by creating a binary image (bi_image) based on a threshold value.
# The thresholded image is then used as a mask to extract the corresponding pixels from the spectral cube.
cropped_img = crop_cube[60:630, 40:462, 29]
cropped_cube = crop_cube[60:630, 40:462, :149]
bi_image = cv2.threshold(cropped_img, 0.8, 1.05, cv2.THRESH_BINARY)[1]

# Line ranges that specify the regions of interest in the image.
# These ranges are used to extract the spectral responses of the foreground pixels for each ink.
# Iterate over Line ranges and apply the ink masks to the corresponding line cubes to extract the spectral responses of the ink pixels. 
# The ink spectra are stored in the ink_spectra list.
line_ranges = [(0, 37), (45, 85), (95, 135), (142, 182), (190, 230), (238, 278),
               (286, 324), (335, 372), (381, 420), (428, 465), (476, 512), (525, 561)]
n_lines = len(line_ranges)

ink_masks = [bi_image[r[0]:r[1], 0:422] < 0.2 for r in line_ranges]
line_cubes = [cropped_cube[r[0]:r[1], 0:422, :149] for r in line_ranges]

# Get the spectral responses of foreground pixels for each ink
ink_spectra = []
for cube, mask in zip(line_cubes, ink_masks):
    ink_cube = cube * mask[..., None]
    ink_pixels = ink_cube[ink_cube != 0].reshape(-1, cube.shape[-1])
    ink_spectra.append(ink_pixels)

# Apply k-means clustering on the spectral responses
X = np.concatenate(ink_spectra)
kmeans = KMeans(n_clusters=len(ink_spectra), random_state=30, n_init=10)
kmeans.fit(X)
spectral_labels = kmeans.labels_

# Count the unique labels to estimate the number of inks
num_inks_kmeans = len(np.unique(spectral_labels))
print(f"Estimated number of inks from k-means clustering: {num_inks_kmeans}")

import cv2
import numpy as np
import matplotlib.pyplot as plt
import spectral
from sklearn.cluster import KMeans

# Load the hyperspectral image
img = spectral.open_image('input/w01_p02_corrected.hdr').load()

# Load the grayscale image
crop_img = cv2.imread('output/hyperspectral_image_grey_scale.png', cv2.IMREAD_GRAYSCALE)

# now use Spectral Analysis
# Extract the cube corresponding to the cropped image 
# from the hyperspectral image to work with the spectral information
crop_cube = img[60:630, 40:462, :149]

# Apply thresholding to segment the image and extract the foreground pixels
# This is done by creating a binary image (bi_image) based on a threshold value.
# The thresholded image is then used as a mask to extract the corresponding pixels from the spectral cube.
cropped_img = crop_cube[60:630, 40:462, 29]
cropped_cube = crop_cube[60:630, 40:462, :149]
bi_image = cv2.threshold(cropped_img, 0.8, 1.05, cv2.THRESH_BINARY)[1]

# Line ranges that specify the regions of interest in the image.
# These ranges are used to extract the spectral responses of the foreground pixels for each ink.
# Iterate over Line ranges and apply the ink masks to the corresponding line cubes to extract the spectral responses of the ink pixels. 
# The ink spectra are stored in the ink_spectra list.
line_ranges = [(0, 37), (45, 85), (95, 135), (142, 182), (190, 230), (238, 278),
               (286, 324), (335, 372), (381, 420), (428, 465), (476, 512), (525, 561)]
n_lines = len(line_ranges)

ink_masks = [bi_image[r[0]:r[1], 0:422] < 0.2 for r in line_ranges]
line_cubes = [cropped_cube[r[0]:r[1], 0:422, :149] for r in line_ranges]

# Get the spectral responses of foreground pixels for each ink
ink_spectra = []
for cube, mask in zip(line_cubes, ink_masks):
    ink_cube = cube * mask[..., None]
    ink_pixels = ink_cube[ink_cube != 0].reshape(-1, cube.shape[-1])
    ink_spectra.append(ink_pixels)

# Apply k-means clustering on the spectral responses
X = np.concatenate(ink_spectra)
kmeans = KMeans(n_clusters=len(ink_spectra), random_state=30, n_init=10)
kmeans.fit(X)
spectral_labels = kmeans.labels_

# Count the unique labels to estimate the number of inks
num_inks_kmeans = len(np.unique(spectral_labels))
print(f"Estimated number of inks from k-means clustering: {num_inks_kmeans}")

