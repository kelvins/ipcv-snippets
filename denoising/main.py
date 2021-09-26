import argparse
import os

import cv2
import numpy as np
from skimage.util import random_noise


def main(image_path):
    image = cv2.imread(image_path)

    noisy_image = random_noise(image, mode='s&p', amount=0.01)
    noisy_image = random_noise(noisy_image, mode='speckle', mean=0.1)
    noisy_image = np.array(255 * noisy_image, dtype='uint8')

    denoised_image = cv2.fastNlMeansDenoisingColored(noisy_image, None, 10, 10, 7, 21)

    result = np.concatenate((image, noisy_image, denoised_image), axis=1)
    cv2.imshow('result', result)
    cv2.waitKey(0)


if __name__ == '__main__':
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    arguments = argparse.ArgumentParser()
    arguments.add_argument(
        '-i',
        '--image',
        type=str,
        default=os.path.join(curr_dir, '..', 'images', 'lenna.png'),
        help='path to input image where we\'ll apply denoising',
    )
    parsed = arguments.parse_args()
    main(parsed.image)
