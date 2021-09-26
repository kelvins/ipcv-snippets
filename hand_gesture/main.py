import math

import cv2
import mediapipe as mp


# Positions:
# https://google.github.io/mediapipe/solutions/hands#hand-landmark-model
def find_positions(frame, landmarks):
    positions = list()
    for index, landmark in enumerate(landmarks.landmark):
        height, width, _ = frame.shape
        cx, cy = int(landmark.x * width), int(landmark.y * height)
        positions.append((index, cx, cy))
    return positions


def main(velocity=15):
    capture = cv2.VideoCapture(0)

    if not capture.isOpened():
        raise Exception('Error opening video capture')

    width = capture.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)

    hands = mp.solutions.hands.Hands()

    shots = list()
    prepared = False
    while True:
        rval, frame = capture.read()

        if not rval:
            break

        results = hands.process(frame)

        for handmarks in results.multi_hand_landmarks or []:
            # Get the position of all fingers
            positions = find_positions(frame, handmarks)

            if len(positions) > 8:
                # Get the position of the thumb
                _, x1, y1 = positions[4]
                # Get the position of the forefinger
                _, x2, y2 = positions[8]

                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                cv2.circle(frame, (x1, y1), 10, (255, 0, 255), cv2.FILLED)
                cv2.circle(frame, (x2, y2), 10, (255, 0, 255), cv2.FILLED)
                cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)

                # Calculate the distance between the forefinger and the thumb
                length = math.hypot(x2 - x1, y2 - y1)

                if length < 50:
                    cv2.circle(frame, (cx, cy), 10, (255, 0, 255), cv2.FILLED)
                    prepared = True
                else:
                    if prepared:
                        shots.append([cx, cy])
                        prepared = False
                    cv2.circle(frame, (cx, cy), 10, (0, 255, 0), cv2.FILLED)

        # Update the position and print the circle
        for shot in shots:
            shot[0] += velocity
            shot[1] -= velocity
            cv2.circle(frame, shot, 10, (255, 0, 0), cv2.FILLED)

        # Remove shots that are outside the boundaries
        shots = [shot for shot in shots if 0 < shot[0] < width and 0 < shot[1] < height]

        cv2.imshow('Hand Gesture', frame)

        if cv2.waitKey(20) == ord('q'):
            break

    cv2.destroyAllWindows()
    capture.release()


if __name__ == '__main__':
    main()
