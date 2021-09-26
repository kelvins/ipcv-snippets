import argparse
import os

import cv2


def main(image_path):
    image = cv2.imread(image_path)

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Blur the image to facilitate the edge detection
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

    sobel_edges = cv2.Sobel(blurred_image, cv2.CV_64F, 1, 1, 5)

    canny_edges = cv2.Canny(blurred_image, 50, 150)

    cv2.imshow('Original', image)
    cv2.imshow('Gray Image', gray_image)
    cv2.imshow('Blurred Image', blurred_image)
    cv2.imshow('Sobel', sobel_edges)
    cv2.imshow('Canny', canny_edges)
    cv2.waitKey(0)


if __name__ == '__main__':
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    arguments = argparse.ArgumentParser()
    arguments.add_argument(
        '-i',
        '--image',
        type=str,
        default=os.path.join(curr_dir, '..', 'images', 'link.png'),
        help='path to input image where we\'ll apply edge detection',
    )
    parsed = arguments.parse_args()
    main(parsed.image)
