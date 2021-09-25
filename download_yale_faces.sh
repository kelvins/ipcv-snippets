#!/bin/bash
wget http://vision.ucsd.edu/datasets/yale_face_dataset_original/yalefaces.zip
unzip -q yalefaces.zip "yalefaces/*" -d ./images
rm yalefaces.zip
