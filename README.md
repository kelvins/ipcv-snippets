# IPCV Snippets

A collection of Image Processing and Computer Vision code snippets.

| Snippets                                                    | Description                                           |
|-------------------------------------------------------------|-------------------------------------------------------|
| [WebCam](./webcam/main.py)                                  | Accessing webcam frames using OpenCV.                 |
| [ImageGrab](./image_grab/main.py)                           | Take a screenshot, save it as PNG and show the image. |
| [Resizing](./resizing/main.py)                              | Resize an image using OpenCV resizing methods.        |
| [Smoothing](./smoothing/main.py)                            | Smooth an image using OpenCV filters.                 |
| [Filter2D](./filter2d/main.py)                              | Apply filter to an image using predefined kernels.    |
| [Watermark](./watermark/main.py)                            | Add watermark to images with OpenCV.                  |
| [Youtube](./youtube/main.py)                                | Load a Youtube video and process each frame.          |
| [EdgeDetection](./edge_detection/main.py)                   | Edge detection using OpenCV (Canny and Sobel).        |
| [ImageHashing](./image_hashing/main.py)                     | Calculate image hashing and distance hashing (dhash). |
| [ExtractColors](./extract_colors/main.py)                   | Given an input image, extract the main colors.        |
| [SuperResolution](./super_resolution/main.py)               | Super resolution using deep learning on OpenCV.       |
| [FaceDetection](./face_detection/main.py)                   | Face detection using cascade classifiers.             |
| [AnonymizeFaces](./anonymize_faces/main.py)                 | Anonymize people's faces by blurring them.            |
| [LongExposure](./long_exposure/main.py)                     | Given a video input creates a long exposure effect.   |
| [ForegroundSegmentation](./foreground_segmentation/main.py) | Foreground segmentation and extraction with GrabCut.  |
| [TemplateMatching](./template_matching/main.py)             | Multi-template matching using OpenCV.                 |
| [FaceRecognition (LBPH)](./lbph/main.py)                    | Face recognition using LBPH algorithm.                |
| [HandTracking](./hand_tracking/main.py)                     | Hand tracking using OpenCV and Media Pipe.            |
| [HandGesture](./hand_gesture/main.py)                       | Shoot balls using your fingers.                       |

## Datasets

To run the face recognition algorithms we need to download the [Yale Face Database](http://vision.ucsd.edu/content/yale-face-database):

```shell
$ bash download_yale_faces.sh
```

## References

- [Advanced Computer Vision with Python - Full Course](https://www.youtube.com/watch?v=01sAkU_NvOY)
- [OpenCV Documentation](https://docs.opencv.org/4.5.3/)
- [OpenCV Forum](https://forum.opencv.org/)
- [PyImageSearch](https://www.pyimagesearch.com/)
