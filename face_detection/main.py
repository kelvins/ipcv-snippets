# Classifiers can be found here:
# https://github.com/opencv/opencv/tree/master/data
import os

import cv2


def main(classifier_path, image_path):
    classifier = cv2.CascadeClassifier(classifier_path)
    image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = classifier.detectMultiScale(
        gray_image,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE,
    )

    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow('Faces', image)
    cv2.waitKey(0)


if __name__ == '__main__':
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    classifier_path = os.path.join(
        curr_dir, 'classifiers', 'haarcascade_frontalface_default.xml'
    )
    image_path = os.path.join(curr_dir, '..', 'images', 'lenna.png')
    main(classifier_path, image_path)
