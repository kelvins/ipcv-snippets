import argparse
import os

import cv2
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split


def load_dataset(dataset_path):
    images = list()
    targets = list()
    for dirpath, _, filenames in os.walk(dataset_path):
        for filename in filenames:
            if filename.startswith('subject'):
                path = os.path.join(dirpath, filename)
                # Load image using PIL and convert to np.array
                # (opencv does not support GIF format)
                image = np.array(Image.open(path))
                images.append(image)
                # Convert subject string to integer
                targets.append(int(filename.split('.')[0][-2:]))
    return images, targets


def detect_faces(
    classifier,
    images,
    targets,
    scale_factor=1.1,
    min_neighbors=5,
    min_size=(30, 30),
    flags=cv2.CASCADE_SCALE_IMAGE,
):
    faces_detected = list()
    faces_targets = list()
    for image, target in zip(images, targets):
        faces = classifier.detectMultiScale(
            image,
            scaleFactor=scale_factor,
            minNeighbors=min_neighbors,
            minSize=min_size,
            flags=flags,
        )

        for (x, y, w, h) in faces:
            crop_img = image[y : y + h, x : x + w]
            faces_detected.append(crop_img)
            faces_targets.append(target)

    return faces_detected, faces_targets


def recognize_faces(recognizer, faces, targets):
    counter = 0
    correct = 0
    for image, target in zip(faces, targets):
        label, threshold = recognizer.predict(image)
        text = f'Subject {label} - Threshold {threshold:.2f}'
        cv2.imshow(text, image)

        # Here we're comparing the label to the target to check if
        # the algorithm is predicting correctly. In a real environment
        # we could use a threshold to assume the prediction is correct.
        counter += 1
        if label == target:
            correct += 1

        key = cv2.waitKey(0)
        cv2.destroyWindow(text)
        if key == ord('q'):
            break

    cv2.destroyAllWindows()

    print(f'Total Faces: {counter}')
    print(f'Correctly Recognized Faces: {correct}')
    print(f'Incorrectly Recognized Faces: {counter-correct}')


def show_subjects(images, targets):
    subjects = dict()
    for image, target in zip(images, targets):
        if target in subjects:
            continue
        img = image.copy()
        cv2.putText(
            img,
            f'Subject {target}',
            (20, 20),
            fontFace=cv2.FONT_HERSHEY_COMPLEX,
            fontScale=0.6,
            color=(0, 255, 0),
        )
        height, width = img.shape[:2]
        img = cv2.resize(
            img, (int(width * 0.6), int(height * 0.6)), interpolation=cv2.INTER_CUBIC
        )
        subjects[target] = img

    rows = list()
    row = list()
    for key in sorted(subjects.keys()):
        row.append(subjects[key])
        if len(row) == 5:
            rows.append(np.concatenate(row, axis=1))
            row = list()

    subjects = np.concatenate(rows, axis=0)
    cv2.imshow('Subjects', subjects)


def main(dataset_path, classifier_path, test_size=0.2, random_state=42):
    # Load the dataset
    images, targets = load_dataset(dataset_path)

    # Create the face detector object
    classifier = cv2.CascadeClassifier(classifier_path)

    # Create the face recognizer object
    face_recognizer = cv2.face.LBPHFaceRecognizer_create(
        radius=1, neighbors=8, grid_x=8, grid_y=8
    )

    # Detect all faces and crop
    faces, targets = detect_faces(classifier, images, targets)

    # Show each subject with its label
    show_subjects(images, targets)

    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(
        faces, targets, test_size=test_size, random_state=random_state
    )

    face_recognizer.train(X_train, np.array(y_train))

    # Recognize the faces in the test set
    recognize_faces(face_recognizer, X_test, y_test)


if __name__ == '__main__':
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    arguments = argparse.ArgumentParser()
    arguments.add_argument(
        '-d',
        '--dataset',
        type=str,
        default=os.path.join(curr_dir, '..', 'images', 'yalefaces'),
        help='path to the faces dataset',
    )
    arguments.add_argument(
        '-c',
        '--classifier',
        type=str,
        default=os.path.join(
            curr_dir, 'classifiers', 'haarcascade_frontalface_default.xml'
        ),
        help='path to the face detection classifier',
    )
    parsed = arguments.parse_args()
    main(parsed.dataset, parsed.classifier)
