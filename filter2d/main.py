import argparse
import os

import cv2
import numpy as np


def main(image_path, kernel_path):
    original = cv2.imread(image_path)
    kernel = np.loadtxt(kernel_path)
    print(f'[INFO] Kernel:\n{kernel}')
    filtered = cv2.filter2D(original, 0, kernel)
    cv2.imshow('Results', np.concatenate((original, filtered), axis=1))
    print('[INFO] Press q to quit')
    cv2.waitKey(0)


if __name__ == '__main__':
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    arguments = argparse.ArgumentParser()
    arguments.add_argument(
        '-i',
        '--image',
        type=str,
        default=os.path.join(curr_dir, '..', 'images', 'lenna.png'),
        help='path to input image where we\'ll apply the filter',
    )
    arguments.add_argument(
        '-k',
        '--kernel',
        type=str,
        default=os.path.join(curr_dir, 'kernels', 'kernel1.txt'),
        help='path to a text file containing the kernel',
    )
    parsed = arguments.parse_args()
    main(parsed.image, parsed.kernel)
