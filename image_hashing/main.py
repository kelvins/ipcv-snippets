import argparse
import os

import cv2


def dhash(image, hash_size=8):
    resized = cv2.resize(image, (hash_size + 1, hash_size))
    diff = resized[:, 1:] > resized[:, :-1]
    return ''.join(diff.flatten().astype(int).astype(str))


def hamming_distance(hash1, hash2):
    if len(hash1) != len(hash2):
        raise ValueError('Hash have different sizes')
    return sum([1 for elem1, elem2 in zip(hash1, hash2) if elem1 != elem2])


def main(image_path1, image_path2, hash_size):
    image1 = cv2.imread(image_path1)
    image2 = cv2.imread(image_path2)
    hash1 = dhash(image1, hash_size=hash_size)
    hash2 = dhash(image2, hash_size=hash_size)
    distance = hamming_distance(str(hash1), str(hash2))
    print('Hash 1:', hash1)
    print('Hash 2:', hash2)
    print('Distance:', distance)


if __name__ == '__main__':
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    arguments = argparse.ArgumentParser()
    arguments.add_argument(
        '--image1',
        type=str,
        default=os.path.join(curr_dir, '..', 'images', 'lenna.png'),
    )
    arguments.add_argument(
        '--image2', type=str, default=os.path.join(curr_dir, '..', 'images', 'link.png')
    )
    arguments.add_argument('--hash_size', type=int, default=8)
    parsed = arguments.parse_args()
    main(parsed.image1, parsed.image2, parsed.hash_size)
