import cv2
import numpy as np

# Load the image
image = cv2.imread('./lenna_full.jpg',0)

# Resize the image
image = cv2.resize(image, (0, 0), None, 0.5, 0.5)

# Gaussian Filter
def gaussian_filter(image, size=3, sigma=1):
    # Create a Gaussian kernel
    kernel = np.zeros((size, size))
    center = size // 2
    total = 0
    for i in range(size):
        for j in range(size):
            x = i - center
            y = j - center
            kernel[i, j] = np.exp(-(x**2 + y**2) / (2 * sigma**2))
            total += kernel[i, j]

    # Normalize the kernel
    kernel /= total

    # Apply the filter
    output = np.zeros_like(image)
    height, width = image.shape
    for i in range(center, height - center):
        for j in range(center, width - center):
            patch = image[i - center: i + center + 1, j - center: j + center + 1]
            output[i, j] = np.sum(patch * kernel)

    cv2.imshow('Gaussian Filter Image', output)
    return output