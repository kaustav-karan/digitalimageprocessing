import cv2
import numpy as np

# Load the image
image = cv2.imread('./lenna_full.jpg',0)

# Resize the image
image = cv2.resize(image, (0, 0), None, 0.5, 0.5)

# Laplacian of Gaussian
def laplacian_of_gaussian(image, size=3):
    output = cv2.GaussianBlur(image, (size, size), 0)
    output = cv2.Laplacian(output, cv2.CV_64F)
    cv2.imshow('Laplacian of Gaussian Image', output)
    return output

# Laplacian Filter
def laplacian_filter(image):
    kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    output = cv2.filter2D(image, -1, kernel)
    cv2.imshow('Laplacian Filter Image', output)
    return output

# Unsharp Masking
def unsharp_masking(image, alpha=1.5):
    blurred = cv2.GaussianBlur(image, (0, 0), 1.5)
    output = cv2.addWeighted(image, 1.5, blurred, -0.5, 0)
    cv2.imshow('Unsharp Masking Image', output)
    return output

unsharp_masking(laplacian_filter(image))



cv2.waitKey(0)

# Display the image
cv2.imshow('Original Image', image)