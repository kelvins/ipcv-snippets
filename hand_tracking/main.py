import cv2
import mediapipe as mp


def main():
    cv2.namedWindow('Hand Tracking')
    capture = cv2.VideoCapture(0)

    if not capture.isOpened():
        raise Exception('Error opening video capture')

    hands = mp.solutions.hands.Hands()

    while True:
        rval, frame = capture.read()

        if not rval:
            break

        results = hands.process(frame)

        for hand_landmarks in results.multi_hand_landmarks or []:
            mp.solutions.drawing_utils.draw_landmarks(
                frame, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS
            )

        cv2.imshow('Hand Tracking', frame)

        if cv2.waitKey(20) == ord('q'):
            break

    cv2.destroyWindow('preview')
    capture.release()


if __name__ == '__main__':
    main()
