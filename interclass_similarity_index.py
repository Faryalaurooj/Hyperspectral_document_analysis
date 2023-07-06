import numpy as np
from sklearn.cluster import KMeans
import cv2

# ... existing code ...

# Load the segmented image for K-means clustering from the output folder
segmented_image_kmeans = cv2.imread('output/segmented_image_kmeans.jpeg')

# Load the segmented image for Gaussian Mixture Model from the output folder
segmented_image_gmm = cv2.imread('output/segmented_image_gmm.jpeg')

# Get the shape of the segmented images for K-means
height, width, channels = segmented_image_kmeans.shape
# Reshape the image to 2D array (height x width, channels)
image_2d_kmeans = np.reshape(segmented_image_kmeans, (height * width, channels))

# Get the shape of the segmented images for Gaussian Mixture Model
height, width, channels = segmented_image_gmm.shape
# Reshape the image to 2D array (height x width, channels)
image_2d_gmm = np.reshape(segmented_image_gmm, (height * width, channels))

# Apply k-means clustering
k = 6  # Number of clusters
kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
kmeans.fit(image_2d_kmeans)

# Get the cluster labels and cluster centers
labels_kmeans = kmeans.labels_
centers_kmeans = kmeans.cluster_centers_
num_inks_kmeans = len(np.unique(labels_kmeans))
print("Number of different inks detected (K-means):", num_inks_kmeans)

# ... existing code ...

# Calculate interclass similarity index
def interclass_similarity_index(segmented_image):
    num_pixels = segmented_image.size
    unique_labels = np.unique(segmented_image)

    similarity_index_sum = 0
    pair_count = 0

    for i in range(len(unique_labels) - 1):
        label_1 = unique_labels[i]
        pixels_1 = segmented_image[segmented_image == label_1]

        for j in range(i + 1, len(unique_labels)):
            label_2 = unique_labels[j]
            pixels_2 = segmented_image[segmented_image == label_2]

            intersection = np.sum(pixels_1 == pixels_2)
            union = len(pixels_1) + len(pixels_2) - intersection

            similarity_index_sum += intersection / union
            pair_count += 1

    interclass_similarity = similarity_index_sum / pair_count
    return interclass_similarity

# Call the function to calculate interclass similarity for K-means segmentation
interclass_similarity_kmeans = interclass_similarity_index(labels_kmeans)

# ... existing code ...


