import argparse
from operator import itemgetter

import cv2
import imutils
import numpy as np


def max_contour(image):
    # Detect edges using Canny and use erosion and dilation to reduce noise
    img = cv2.erode(cv2.dilate(cv2.Canny(image, 10, 200), None), None)
    # Use findContours to detect the contours in the image
    contours, _ = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    # Calc all contour areas and store it in a list of tuple
    contour_areas = [(c, cv2.contourArea(c)) for c in contours]
    # Return the maximum area from contour_areas
    return max(contour_areas, key=itemgetter(1))[0]


def extract_colors(image, palette):
    # Get the maximum contour from the image
    contour = max_contour(image)
    # Create an array of zeros with the same shape as the image
    mask = np.zeros_like(image)
    # Change the mask by applying a white contour
    cv2.fillConvexPoly(mask, contour, (255,) * image.shape[2])
    # Apply the mask to the image
    image = image[np.nonzero(mask)[:2]]
    # Use kmeans to find the centroids that represents the predominant
    # colors. Note that palette is the K from kmeans
    _, _, centroids = cv2.kmeans(
        image.astype(np.float32),
        palette,
        None,
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0),
        100,
        cv2.KMEANS_PP_CENTERS,
    )
    # Return all centroids converting from GBR to RGB
    return [tuple(map(int, centroid[::-1])) for centroid in centroids]


def show_results(image, colors, width=800):
    # Resize the image for better visualization
    output = imutils.resize(image, width=width)
    # Calculate the color boxes size based on the image width
    h, w = 100, int(width / len(colors))
    # Create the color pallete
    palette = np.zeros((h, width, 3), np.uint8)
    # Draw all color boxes
    for i, color in enumerate(colors):
        cv2.rectangle(palette, (i * w, 0), (i * w + w, h), color[::-1], -1)
    cv2.imshow('Results', np.concatenate((output, palette), axis=0))
    cv2.waitKey(0)


def main(image_path, palette):
    image = cv2.imread(image_path)
    colors = extract_colors(image, palette)
    print(colors)
    show_results(image, colors)


if __name__ == '__main__':
    arguments = argparse.ArgumentParser()
    arguments.add_argument('--image', required=True)
    arguments.add_argument('--palette', type=int, default=5)
    parsed = arguments.parse_args()
    main(parsed.image, parsed.palette)
