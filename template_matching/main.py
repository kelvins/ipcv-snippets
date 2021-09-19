import os

import numpy as np
import argparse
import cv2


def non_max_suppression(boxes, threshold=0.3):
    if not boxes.any():
        return list()

    # If the bounding boxes are integers, convert them to floats
    if boxes.dtype.kind == 'i':
        boxes = boxes.astype('float')

    # Grab the coordinates of the bounding boxes
    x1, y1, x2, y2 = boxes[:,0], boxes[:,1], boxes[:,2], boxes[:,3]

    # Compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    indexes = np.argsort(y2)

    pick = list()
    while indexes.any():
        last = len(indexes) - 1
        index = indexes[last]
        pick.append(index)

        # Find the largest (x, y) coordinates for the start of the bounding box
        # and the smallest (x, y) coordinates for the end of the bounding box
        xx1 = np.maximum(x1[index], x1[indexes[:last]])
        yy1 = np.maximum(y1[index], y1[indexes[:last]])
        xx2 = np.minimum(x2[index], x2[indexes[:last]])
        yy2 = np.minimum(y2[index], y2[indexes[:last]])

        width = np.maximum(0, xx2 - xx1 + 1)
        height = np.maximum(0, yy2 - yy1 + 1)

        # Compute the ratio of overlap
        overlap = (width * height) / area[indexes[:last]]
        indexes = np.delete(
            indexes,
            np.concatenate(([last], np.where(overlap > threshold)[0]))
        )

    # Return only the bounding boxes that were picked as integers
    return boxes[pick].astype('int')


def main(image_path, template_path, threshold):
    image = cv2.imread(image_path)
    template = cv2.imread(template_path)

    template_height, template_width = template.shape[:2]

    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

    print('[INFO] Performing template matching...')
    result = cv2.matchTemplate(image_gray, template_gray, cv2.TM_CCOEFF_NORMED)

    # Find all locations where the matched value is greater than the threshold
    y_coords, x_coords = np.where(result >= threshold)
    print(f'[INFO] {len(y_coords)} matched locations before NMS')

    rects = []
    for x, y in zip(x_coords, y_coords):
        rects.append((x, y, x + template_width, y + template_height))

    # Apply non-maxima suppression to the rectangles
    pick = non_max_suppression(np.array(rects))
    print(f'[INFO] {len(pick)} matched locations after NMS')

    for start_x, start_y, end_x, end_y in pick:
        cv2.rectangle(image, (start_x, start_y), (end_x, end_y), (255, 0, 0), 2)

    cv2.imshow('Matchings', image)
    cv2.waitKey(0)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument(
        '-i', '--image', type=str,
        default=os.path.join(curr_dir, '..', 'images', 'space_invaders.png'),
        help='path to input image where we\'ll apply template matching'
    )
    ap.add_argument(
        '-t', '--template', type=str,
        default=os.path.join(curr_dir, '..', 'images', 'space_invaders_template.png'),
        help='path to template image'
    )
    ap.add_argument(
        '-b', '--threshold', type=float, default=0.8,
        help='threshold for multi-template matching'
    )
    args = vars(ap.parse_args())
    main(args['image'], args['template'], args['threshold'])
