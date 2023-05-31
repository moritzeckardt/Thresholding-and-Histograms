# Implement the histogram equalization in this file
import cv2
import numpy as np

# Load the image
image = cv2.imread('hello.png', cv2.IMREAD_GRAYSCALE)

# Compute the histogram
histogram = np.bincount(image.flatten(), minlength=256)

# Compute the cumulative distribution function (CDF)
cdf = np.cumsum(histogram)

# Normalize the CDF
cdf_normalized = (cdf - cdf.min()) / (cdf.max() - cdf.min())

# Apply histogram equalization to each pixel
height, width = image.shape
total_pixels = height * width
for i in range(height):
    for j in range(width):
        old_pixel_value = image[i, j]
        new_pixel_value = int(cdf_normalized[old_pixel_value] * 255)
        image[i, j] = new_pixel_value

# Save the result
cv2.imwrite('kitty.png', image)
