import spectral
import numpy as np
import matplotlib.pyplot as plt

# Load the hyperspectral image using the .hdr file
img = spectral.open_image('input/w01_p02_corrected.hdr')

# Select the bands wanted to be rescaled
bands = [0, 29, 59, img.shape[2]-1]

# Loop through the selected bands and rescale the values
for i, band in enumerate(bands):
    # Extract the band from the hyperspectral image
    img_band = img.read_band(band)
    plt.subplot(2, 2, i+1)
    plt.imshow(img[:,:,band], cmap='gray', vmin=0, vmax=1)
    plt.title('Band {}'.format(band+1))

# Show the plot
plt.show()



