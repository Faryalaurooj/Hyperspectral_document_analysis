# Hyperspectral_document_analysis
In this repo, we will explore the strength of hyper spectral image analysis for identification of ink mismatch in a document
A hyperspectral cube from iVision HHID dataset is provided here. The document contains some handwritten text written using one or more pens of different brands. We were required to perform the following tasks:
a) Show the total number of bands available in the hyperspectral cube along with the starting and ending wavelength range.
b) Display the 1st, 30th, 60th and the last band of the hyperspectral image as separate grayscale images. (Total 4 images). Also explain the visual differences that you find in the greyscale images.
c) Plot the spectral responses of foreground (or text) pixels only against the wavelength or band numbers. (Wavelengths on x-axis, Reflectance of text pixels on y-axis). Explain the trend in the graph.
d) Use an appropriate pattern recognition method to detect the number of different inks present in the document. Discuss the strengths and weakness / limitations of your approach.
e) Use color-labeling to classify text written with different inks in the document.

Solution:
(a) We first looked into the shape of the hyperspectral image and total no of spectral bands with the help of code mentioned as (a) given in files. So, it was found that the image is a tuple (650, 512, 149) which means that the image has 650 rows, 512 columns, and 149 spectral bands.  In hyperspectral imaging, each pixel in the image is associated with a spectrum of values, instead of just a single-color value as in RGB images. The spectral bands in this case represent the different wavelengths of light that were captured by the hyperspectral sensor. Therefore, the image with shape (650, 512, 149) is a hyperspectral image with 650 rows, 512 columns, and 149 spectral bands.
Then, we load the hyperspectral image and use the bands attribute of the img object to get the total number of bands in the hyperspectral cube. Finally, we extract the starting and ending wavelength range from the metadata dictionary of the img object using the wavelength key. 
Output of code is: - 
Starting wavelength:  478.7825462 nm 
Ending wavelength:  900.9723394  nm
Each hyperspectral cube in the dataset contains 149 spectral channels in the spectral range of 478-901 nm. Resolution= (901-478) / 149 = 2.89 nm between two channels.

(b) In order to Display the 1st, 30th, 60th and the last band of the hyperspectral image as separate grayscale images. (Total 4 images). Also explain the visual differences that you find in the greyscale images. First of all, the hyperspectral image is displayed as a grey scale image on laptop screen with the help of this code mentioned as (b). In this code, we have loaded the hyperspectral image. The vmin and vmax parameters are set to the 5th and 95th percentiles of the data range, respectively to fine-tune the brightness and contrast of the image. The output of this code shows this greyscale version of the hyperspectral image provided to us.
In step 2, we tried to display the 1st, 30th, 60th and the last band of the hyperspectral image as separate grayscale images (Total 4 images) with the help of this code.

(c) In order to Plot the spectral responses of foreground (or text) pixels only against the wavelength or band numbers. (Wavelengths on x-axis, Reflectance of text pixels on y-axis). Explain the trend in the graph. We write code (c). This code demonstrates how to extract spectral information  for different inks from a hyperspectral image, visualize the spectral  responses and analyze the reflectance characteristics at different wavelengths in single 2d plot.
The code performs following functions: -
(1) Loads the hyperspectral image using the spectral library.
(2)	It checks if the wavelength information is present in the metadata dictionary of the image. If not, it tries to read the wavelength information from the header file.
(3)	It extracts a mask for the text pixels in the image.
(4)	It extracts the spectral responses of the text pixels by using the mask and computes the spectrum of the text pixels for each ink type.
(5)	Finally, it plots the spectrum of the text pixels against the wavelengths of each ink type on the x-axis and reflectance on the y-axis.

(d) In order to Use an appropriate pattern recognition method to detect the number of different inks present in the document. Discuss the strengths and weakness / limitations of your approach. We write code mentioned as (d). Code applies k-means clustering on the spectral response (wavelengths vs reflectance plot) of ink pixels extracted from a hyperspectral image. The number of ink clusters is estimated based on the spectral responses using the elbow method and Silhoutte coefficient.

(e) In order to Use color-labeling to classify text written with different inks in the document. We merged codes (d) and (e) mentioned as (d, e together) in files. The code will display the ink mask obtained by applying thresholding to segment the image and extract the foreground pixels. It will run K-means clustering on the selected (foreground/ink) pixels using K=6. The code will then plot the inertia values and silhouette scores to help determine the appropriate number of clusters (inks). The code will display the segmented image using K-means clustering, with each cluster labeled with a different color. The number of different inks detected will also be printed. code will perform Gaussian mixture modeling on the selected pixels. It will plot the segmented image obtained from the Gaussian mixture model, with each cluster labeled with a different color. It will plot the segmented image with color-labeling using K-means clustering. It will plot the segmented image with color-labeling using Gaussian mixture modeling.


Hope you like the work
feel free to post any querries and suggesstions as well :)




