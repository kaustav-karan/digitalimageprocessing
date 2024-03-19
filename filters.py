import cv2
import numpy as np

# Load the image
image = cv2.imread('./lenna_full.jpg',0)

# Resize the image
image = cv2.resize(image, (0, 0), None, 0.5, 0.5)

# Gemetric Mean Filter
def geometric_mean_filter(image, size=3):
    temp = np.zeros(image.shape, np.uint8)
    border = size // 2
    for i in range(border, image.shape[0] - border):
        for j in range(border, image.shape[1] - border):
            window = image[i - border:i + border + 1, j - border:j + border + 1]
            window = window.flatten()
            product = 1
            for pixel in window:
                product *= pixel
            temp[i][j] = product ** (1 / (size ** 2))
    cv2.imshow('Geometric Mean Filter Image', temp)
    return temp

# Harmonic Mean Filter
def harmonic_mean_filter(image, size=3):
    temp = np.zeros(image.shape, np.uint8)
    border = size // 2
    for i in range(border, image.shape[0] - border):
        for j in range(border, image.shape[1] - border):
            window = image[i - border:i + border + 1, j - border:j + border + 1]
            window = window.flatten()
            total = 0
            for pixel in window:
                total += 1 / pixel
            temp[i][j] = (size ** 2) / total
    cv2.imshow('Harmonic Mean Filter Image', temp)
    return temp

# Contraharmonic Mean Filter
def contraharmonic_mean_filter(image, size=3, Q=1):
    temp = np.zeros(image.shape, np.uint8)
    border = size // 2
    for i in range(border, image.shape[0] - border):
        for j in range(border, image.shape[1] - border):
            window = image[i - border:i + border + 1, j - border:j + border + 1]
            window = window.flatten()
            num = 0
            den = 0
            for pixel in window:
                num += pixel ** (Q + 1)
                den += pixel ** Q
            temp[i][j] = num / den
    cv2.imshow('Contraharmonic Mean Filter Image', temp)
    return temp

# Order-Statistics Filter
def order_statistics_filter(image, size=3, order=5):
    temp = np.zeros(image.shape, np.uint8)
    border = size // 2
    for i in range(border, image.shape[0] - border):
        for j in range(border, image.shape[1] - border):
            window = image[i - border:i + border + 1, j - border:j + border + 1]
            window = window.flatten()
            window.sort()
            temp[i][j] = window[order]
    cv2.imshow('Order-Statistics Filter Image', temp)
    return temp

# Arithmetic Mean Filter
def arithmetic_mean_filter(image, size=3):
    temp = np.zeros(image.shape, np.uint8)
    border = size // 2
    for i in range(border, image.shape[0] - border):
        for j in range(border, image.shape[1] - border):
            window = image[i - border:i + border + 1, j - border:j + border + 1]
            temp[i][j] = np.mean(window)
    cv2.imshow('Arithmetic Mean Filter Image', temp)
    return temp

# Midpoint Filter
def midpoint_filter(image, size=3):
    temp = np.zeros(image.shape, np.uint8)
    border = size // 2
    for i in range(border, image.shape[0] - border):
        for j in range(border, image.shape[1] - border):
            window = image[i - border:i + border + 1, j - border:j + border + 1]
            temp[i][j] = (np.max(window) + np.min(window)) // 2
    cv2.imshow('Midpoint Filter Image', temp)
    return temp

# Alpha-Trimmed Mean Filter
def alpha_trimmed_mean_filter(image, size=3, d=2):
    temp = np.zeros(image.shape, np.uint8)
    border = size // 2
    for i in range(border, image.shape[0] - border):
        for j in range(border, image.shape[1] - border):
            window = image[i - border:i + border + 1, j - border:j + border + 1]
            window = window.flatten()
            window.sort()
            window = window[d:-d]
            temp[i][j] = np.mean(window)
    cv2.imshow('Alpha-Trimmed Mean Filter Image', temp)
    return temp
