import os

import cv2
import numpy as np

curr_dir = os.path.dirname(os.path.abspath(__file__))
image_path = os.path.join(curr_dir, '..', 'images', 'lenna.png')
image = cv2.imread(image_path)

# Apply filters
kernel = np.ones((6, 6), np.float32) / 25
filter2D = cv2.filter2D(image, -1, kernel)
blur = cv2.blur(image, (5, 5))
gaussianBlur = cv2.GaussianBlur(image, (5, 5), 0)
median = cv2.medianBlur(image, 5)
bilateralFilter = cv2.bilateralFilter(image, 9, 75, 75)

# Set text to each image
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(image, 'Original Image', (120, 480), font, 1, (0, 0, 255), 2, cv2.LINE_AA)
cv2.putText(filter2D, 'Filter 2D', (220, 480), font, 1, (0, 0, 255), 2, cv2.LINE_AA)
cv2.putText(blur, 'Blur', (220, 480), font, 1, (0, 0, 255), 2, cv2.LINE_AA)
cv2.putText(gaussianBlur, 'Gaussian Blur', (150, 480), font, 1, (0, 0, 255), 2, cv2.LINE_AA)
cv2.putText(median, 'Median', (220, 480), font, 1, (0, 0, 255), 2, cv2.LINE_AA)
cv2.putText(bilateralFilter, 'Bilateral Filter', (150, 480), font, 1, (0, 0, 255), 2, cv2.LINE_AA)

# Concatenate images to output
tempImage1 = np.concatenate((image, median, gaussianBlur), axis=1)
tempImage2 = np.concatenate((bilateralFilter, blur, filter2D), axis=1)
finalImage = np.concatenate((tempImage1, tempImage2), axis=0)

cv2.imshow('Final Image', finalImage)
cv2.waitKey(0)
