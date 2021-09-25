import argparse
import os

import cv2
import numpy as np


def main(image_path, watermark_path, alpha):
    image = cv2.imread(image_path)
    img_h, img_w = image.shape[:2]

    watermark = cv2.imread(watermark_path, cv2.IMREAD_UNCHANGED)
    wat_h, wat_w = watermark.shape[:2]

    image = np.dstack([image, np.ones((img_h, img_w), dtype='uint8') * 255])

    # create an overlay image and add the watermark
    overlay = np.zeros((img_h, img_w, 4), dtype='uint8')
    h = img_h - 10
    w = img_w - 10
    overlay[h - wat_h : h, w - wat_w : w] = watermark

    # blend the two images together using transparent overlays
    cv2.addWeighted(overlay, alpha, image, 1.0, 0, image)

    cv2.imshow('Watermarked', image)
    cv2.waitKey(0)


if __name__ == '__main__':
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    arguments = argparse.ArgumentParser()
    arguments.add_argument(
        '-w',
        '--watermark',
        default=os.path.join(curr_dir, '..', 'images', 'space_invaders_template.png'),
        help='path to watermark image',
    )
    arguments.add_argument(
        '-i',
        '--image',
        type=str,
        default=os.path.join(curr_dir, '..', 'images', 'lenna.png'),
        help='image path to apply watermark',
    )
    arguments.add_argument(
        '-a',
        '--alpha',
        type=float,
        default=0.25,
        help='alpha transparency of the overlay',
    )
    parsed = arguments.parse_args()
    main(parsed.image, parsed.watermark, parsed.alpha)
