import cv2


class LongExposure:
    def __init__(self, video, output_image_path, step=1):
        self.video = video
        self.step = step

    @staticmethod
    def averager():
        count = 0
        total = 0.0
        def average(value):
            nonlocal count, total
            count += 1
            total += value
            return total / count
        return average

    def run(self):
        # Open a pointer to the video file
        stream = cv2.VideoCapture(self.video)

        # Get the total frames to be used by the progress bar
        total_frames = int(stream.get(cv2.CAP_PROP_FRAME_COUNT))

        r, g, b = None, None, None
        r_avg, g_avg, b_avg = self.averager(), self.averager(), self.averager()

        for count in range(total_frames):
            # Split the frame into its respective channels
            _, frame = stream.read()

            if count % self.step == 0 and frame is not None:
                # Get the current RGB
                b_curr, g_curr, r_curr = cv2.split(frame.astype('float'))
                r, g, b = r_avg(r_curr), g_avg(g_curr), b_avg(b_curr)

        # Merge the RGB averages together and write the output image to disk
        avg = cv2.merge([b, g, r]).astype('uint8')
        cv2.imshow('Long Exposure', avg)

        # Release the stream pointer
        stream.release()


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('-v', '--video', help='path to a video file')
    ap.add_argument('-s', '--step', help='step used to get frames', default=1)
    args = vars(ap.parse_args())
    LongExposure(args['video'], args['step']).run()
