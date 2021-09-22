import argparse

import pafy
import cv2


def main(url):
    video = pafy.new(url)
    best = video.getbest(preftype='mp4')

    capture = cv2.VideoCapture(best.url)

    while True:
        grabbed, frame = capture.read()
        if not grabbed:
            break

        # Resize the image
        frame = cv2.resize(frame, (512, 384), interpolation=cv2.INTER_CUBIC)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        cv2.imshow('Original', frame)
        cv2.imshow('RGB Frame', rgb_frame)

        if cv2.waitKey(20) == ord('q'):
            break

    capture.release()


if __name__ == '__main__':
    url = 'https://www.youtube.com/watch?v=8gXpZmQ7j70'
    arguments = argparse.ArgumentParser()
    arguments.add_argument(
        '-u', '--url', type=str, default=url, help='youtube url'
    )
    parsed = arguments.parse_args()
    main(parsed.url)
