import cv2
import numpy as np

# Read the image
image = cv2.imread('ecg_image.jpeg')

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply adaptive thresholding to create a mask for the grid
grid_mask = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 8)

# Apply morphological operations to remove noise and refine the grid mask
grid_mask = cv2.morphologyEx(grid_mask, cv2.MORPH_OPEN, np.ones((5,5), np.uint8))

# Invert the grid mask
grid_mask = cv2.bitwise_not(grid_mask)

# Create a plain white background
white_background = np.ones_like(image) * 255

# Apply the grid mask to the original image to keep the grid and make the background white
result = cv2.bitwise_and(image, image, mask=grid_mask)
result = cv2.add(result, white_background)

# Display the original and modified images
cv2.imshow('Original Image', image)
cv2.imshow('Modified Image', result)
cv2.waitKey(0)
cv2.destroyAllWindows()
