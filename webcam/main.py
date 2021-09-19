import cv2


def main():
    cv2.namedWindow('preview')
    capture = cv2.VideoCapture(0)

    if not capture.isOpened():
        raise Exception('Error opening video capture')

    rval, frame = capture.read()

    while rval:
        cv2.imshow('preview', frame)
        rval, frame = capture.read()
        if cv2.waitKey(20) == ord('q'):
            break

    cv2.destroyWindow('preview')
    capture.release()


if __name__ == '__main__':
    main()
