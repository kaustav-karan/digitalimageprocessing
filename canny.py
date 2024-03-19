import cv2
import numpy as np

# Load the image
image = cv2.imread('./lenna_full.jpg',0)

# Resize the image
image = cv2.resize(image, (0, 0), None, 0.5, 0.5)

# Cannny Edge Detection
def canny_edge_detection(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Calculate gradients using Sobel operator
    gradient_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)

    # Calculate gradient magnitude and direction
    gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
    gradient_direction = np.arctan2(gradient_y, gradient_x)

    # Apply non-maximum suppression to thin out edges
    suppressed = non_maximum_suppression(gradient_magnitude, gradient_direction)

    # Apply double thresholding to detect strong and weak edges
    strong_edges, weak_edges = double_thresholding(suppressed, low_threshold=50, high_threshold=150)

    # Apply edge tracking by hysteresis to connect weak edges to strong edges
    edges = edge_tracking(strong_edges, weak_edges)

    # Return the resulting edges
    return edges

def non_maximum_suppression(gradient_magnitude, gradient_direction):
    # Perform non-maximum suppression to thin out edges
    rows, cols = gradient_magnitude.shape
    suppressed = np.zeros_like(gradient_magnitude)

    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            angle = gradient_direction[i, j]
            mag = gradient_magnitude[i, j]

            # Determine the neighboring pixels based on the gradient direction
            if (0 <= angle < np.pi / 8) or (15 * np.pi / 8 <= angle <= 2 * np.pi):
                prev_pixel = gradient_magnitude[i, j - 1]
                next_pixel = gradient_magnitude[i, j + 1]
            elif (np.pi / 8 <= angle < 3 * np.pi / 8) or (9 * np.pi / 8 <= angle < 11 * np.pi / 8):
                prev_pixel = gradient_magnitude[i - 1, j + 1]
                next_pixel = gradient_magnitude[i + 1, j - 1]
            elif (3 * np.pi / 8 <= angle < 5 * np.pi / 8) or (11 * np.pi / 8 <= angle < 13 * np.pi / 8):
                prev_pixel = gradient_magnitude[i - 1, j]
                next_pixel = gradient_magnitude[i + 1, j]
            else:
                prev_pixel = gradient_magnitude[i - 1, j - 1]
                next_pixel = gradient_magnitude[i + 1, j + 1]

            # Suppress the pixel if it is not the maximum in the neighborhood
            if mag >= prev_pixel and mag >= next_pixel:
                suppressed[i, j] = mag

    return suppressed

def double_thresholding(edges, low_threshold, high_threshold):
    # Perform double thresholding to detect strong and weak edges
    rows, cols = edges.shape
    strong_edges = np.zeros_like(edges)
    weak_edges = np.zeros_like(edges)

    # Iterate through each pixel in the image
    for i in range(rows):
        for j in range(cols):
            pixel = edges[i, j]

            # Check if the pixel value is above the high threshold
            if pixel >= high_threshold:
                strong_edges[i, j] = pixel
            # Check if the pixel value is above the low threshold
            elif pixel >= low_threshold:
                weak_edges[i, j] = pixel

    return strong_edges, weak_edges

def edge_tracking(strong_edges, weak_edges):
    # Perform edge tracking by hysteresis to connect weak edges to strong edges
    rows, cols = strong_edges.shape
    edges = np.zeros_like(strong_edges)

    # Iterate through each pixel in the image
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            if strong_edges[i, j] > 0:
                edges[i, j] = 255
            elif weak_edges[i, j] > 0:
                # Check if any of the neighboring pixels are strong edges
                if strong_edges[i-1:i+2, j-1:j+2].max() > 0:
                    edges[i, j] = 255

    return edges