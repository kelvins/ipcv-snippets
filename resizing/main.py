import argparse
import os

import cv2
import numpy as np


def main(image_path, ratio):
    image = cv2.imread(image_path)
    height, width = image.shape[:2]

    methods = {
        'INTER_NEAREST': cv2.INTER_NEAREST,
        'INTER_LINEAR': cv2.INTER_LINEAR,
        'INTER_AREA': cv2.INTER_AREA,
        'INTER_CUBIC': cv2.INTER_CUBIC,
        'INTER_LANCZOS4': cv2.INTER_LANCZOS4,
    }

    results = []
    for name, method in methods.items():
        print(f'[INFO] Resizing image using {name} method')
        w, h = int(width * ratio), int(height * ratio)
        resized_image = cv2.resize(image, (w, h), interpolation=method)

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(
            resized_image, name, (10, 20), font, 0.5, (0, 0, 255), 1, cv2.LINE_AA
        )
        results.append(resized_image)

    result = np.concatenate(results, axis=1)
    cv2.imshow('Original', image)
    cv2.imshow('Resizing', result)
    cv2.waitKey(0)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument(
        '-i',
        '--image',
        type=str,
        default=os.path.join(curr_dir, '..', 'images', 'owl.png'),
        help='path to input image where we\'ll apply template matching',
    )
    ap.add_argument('-r', '--ratio', type=float, default=2.0, help='image ratio')
    args = vars(ap.parse_args())
    main(args['image'], args['ratio'])
