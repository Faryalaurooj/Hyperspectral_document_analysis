# author Faryal Aurooj 
# This code applies k means clustering on histogram plotted
# for grey scale image and estimates the number of inks used
# based on the intensity values of pixels
 
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Load the grayscale image
crop_img = cv2.imread('output/hyperspectral_image_grey_scale.png', cv2.IMREAD_GRAYSCALE)

# Apply K-means clustering on the histogram values
X = crop_img.reshape(-1, 1)
kmeans = KMeans(n_clusters=12, random_state=30, n_init=10)
kmeans.fit(X)

# Get the unique labels assigned by K-means
hist_labels = kmeans.labels_

# Count the unique labels to estimate the number of inks
num_inks = len(np.unique(hist_labels))
print(f"Estimated number of inks from K-means clustering: {num_inks}")


