
#The code will display the ink mask obtained by applying thresholding to segment the image and extract the foreground pixels.

#It will run K-means clustering on the selected (foreground/ink) pixels using K=6. The code will then plot the inertia values and silhouette scores to help determine the appropriate number of clusters (inks).

#The code will display the segmented image using K-means clustering, with each cluster labeled with a different color. The number of different inks detected will also be printed.

#code will perform Gaussian mixture modeling on the selected pixels. It will plot the segmented image obtained from the Gaussian mixture model, with each cluster labeled with a different color.

#It will plot the segmented image with color-labeling using K-means clustering.

#It will plot the segmented image with color-labeling using Gaussian mixture modeling.


import cv2
import spectral
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.utils import shuffle
from sklearn.mixture import GaussianMixture

# Load the hyperspectral image
img = spectral.open_image('input/w01_p02_corrected.hdr').load()
crop_img = cv2.imread('output/hyperspectral_image_grey_scale.png', cv2.IMREAD_GRAYSCALE)

if img is not None:
    # Extract the cube for the cropped image
    crop_cube = img[60:630, 40:462, :149]

    # Apply thresholding to segment the image and extract the foreground pixels
    cropped_img = crop_cube[60:630, 40:462, 29]
    cropped_cube = crop_cube[60:630, 40:462, :149]
    bi_image = cv2.threshold(cropped_img, 0.8, 1.05, cv2.THRESH_BINARY)[1]
    line_ranges = [(0, 37), (45, 85), (95, 135), (142, 182), (190, 230), (238, 278),
                   (286, 324), (335, 372), (381, 420), (428, 465), (476, 512), (525, 561)]
    n_lines = len(line_ranges)

    ink_masks = [bi_image[r[0]:r[1], 0:422] < 0.2 for r in line_ranges]
    line_cubes = [cropped_cube[r[0]:r[1], 0:422, :149] for r in line_ranges]
    ink_spectra = []
    for cube, mask in zip(line_cubes, ink_masks):
        ink_cube = cube * mask[..., None]
        ink_pixels = ink_cube[ink_cube != 0].reshape(-1, cube.shape[-1])
        ink_spectra.append(ink_pixels)

    # Get the wavelengths
    wavelengths = img.metadata['wavelength']
    wavelengths = img.bands.centers

else:
    print("Error loading image")

# Part d

# Use an appropriate pattern recognition method to find the number of inks present in the document.
# Generating ink mask for all ink pixels available in the document
ink_mask = bi_image < 0.2
plt.imshow(ink_mask, cmap="gray")
plt.show()

# Assigning non-ink (background) pixels value of 0
selection = cropped_cube
selection_modified = np.copy(selection)
selection_modified[~ink_mask] = 1

# Running K-Means Clustering on selected (foreground / ink) pixels for K = 7

data = selection.reshape(-1, 1)

# Reshaping the image into a 2D array of pixels and 3 color values (RGB)
pixel_vals = selection.reshape((-1, 149))

# Convert to float type
data = np.float32(pixel_vals)

# To find appropriate number of K

# Define the range of possible cluster numbers
k_values = range(5, 8)

# Initialize lists to store the inertia values and silhouette scores
inertia_values = []
silhouette_scores = []

# Perform K-means clustering for each value of k
for k in k_values:
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(data)

    # Append the inertia value and silhouette score to the respective lists
    inertia_values.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(data, kmeans.labels_))

# Plot the inertia values and silhouette scores
plt.figure(figsize=(10, 4))

# Plot the inertia values
plt.subplot(1, 2, 1)
plt.plot(k_values, inertia_values, marker='o')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Method')

# Plot the silhouette scores
plt.subplot(1, 2, 2)
plt.plot(k_values, silhouette_scores, marker='o')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Analysis')

plt.tight_layout()
plt.show()

# Part d with K-means

# Reshape the image to 2D array (height x width, channels)
height, width, channels = selection.shape
image_2d = np.reshape(selection, (height * width, channels))

# Normalize the pixel values to the range [0, 1]
image_2d = image_2d.astype(float) / 255.0

# Apply k-means clustering
k = 6  # Number of clusters
kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
kmeans.fit(image_2d)


# Get the cluster labels and cluster centers
labels = kmeans.labels_
centers = kmeans.cluster_centers_
num_inks = len(np.unique(labels))
print("Number of different inks detected:", num_inks)

# Reshape the labels back to the original image shape
segmented_image = np.reshape(labels, (height, width))

# Plot the segmented image
plt.figure(figsize=(8, 6))
plt.imshow(segmented_image, cmap='nipy_spectral')
plt.axis('off')
plt.title('Spectral Classes from K-means')
plt.show()

# Part d with Gaussian mixture

# Reshape the image to 2D array (height x width, channels)
height, width, channels = selection.shape
image_2d = np.reshape(selection, (height * width, channels))

# Normalize the pixel values to the range [0, 1]
image_2d = image_2d.astype(float) / 255.0

mixture_model = GaussianMixture(n_components=6, max_iter=20)
labels = mixture_model.fit_predict(image_2d)
segmented_image1 = labels.reshape(height, width)

plt.imshow(segmented_image1, cmap='nipy_spectral')
plt.axis('off')
plt.show()
plt.contour(segmented_image1, colors='black', linewidths=0.5)

centers = []
for label in np.unique(labels):
    mask = segmented_image1 == label
    center_y, center_x = np.mean(np.where(mask), axis=1)
    centers.append((center_x, center_y))

plt.scatter(*zip(*centers), color='red', marker='x')
plt.show()

print("Segment centers: ", centers)

# Part e

# Generate a color map for each cluster label
colors = plt.cm.get_cmap('nipy_spectral', k)  # You can choose a different color map if desired

# Plot the segmented image with color-labeling
plt.imshow(segmented_image, cmap=colors)
plt.title("Segmented Image with K-means")
plt.colorbar(ticks=range(k))
plt.show()

# Part e

# Generate a color map for each cluster label
colors = plt.cm.get_cmap('nipy_spectral', k)  # You can choose a different color map if desired

# Plot the segmented image with color-labeling
plt.imshow(segmented_image)

