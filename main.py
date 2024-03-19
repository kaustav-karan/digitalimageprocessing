import cv2 
import numpy as np

# Load the image
image = cv2.imread('./lenna_full.jpg',0)

# Resize the image
image = cv2.resize(image, (0, 0), None, 0.5, 0.5)

# Display the image
cv2.imshow('Original Image', image)

# # Convert the image to grayscale
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Salt and Pepper Noise
def salt_pepper_noise(image, salt=0.125, pepper=0.125):
    output = np.zeros(image.shape, np.uint8)
    thres = 1 - salt - pepper
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            r = np.random.random()
            if r < salt:
                output[i][j] = 0
            elif r > salt and r < thres:
                output[i][j] = image[i][j]
            else:
                output[i][j] = 255
    cv2.imshow('Salt and Pepper Noise Image', output)
    return output

# Adaptive Median Filter
def adaptive_median_filter(image, size=3):
    temp = np.zeros(image.shape, np.uint8)
    border = size // 2
    for i in range(border, image.shape[0] - border):
        for j in range(border, image.shape[1] - border):
            window = image[i - border:i + border + 1, j - border:j + border + 1]
            window = window.flatten()
            window.sort()
            median = window[len(window) // 2]
            if median == 0 or median == 255:
                window = window.tolist()
                window.remove(median)
                median = window[len(window) // 2]
            temp[i][j] = median
    cv2.imshow('Adaptive Median Filter Image', temp)
    return temp

adaptive_median_filter(salt_pepper_noise(image))

cv2.waitKey(0)
cv2.destroyAllWindows()
