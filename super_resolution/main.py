# Pre-trained models from:
# https://github.com/opencv/opencv_contrib/tree/master/modules/dnn_superres
import argparse
import os
import time

import cv2


def export_model_name_and_scale(model_path):
    model_name = model_path.split(os.path.sep)[-1].split('_')[0].lower()
    model_scale = model_path.split('_x')[-1]
    model_scale = int(model_scale[: model_scale.find('.')])
    return model_name, model_scale


def main(model_path, image_path):
    model_name, model_scale = export_model_name_and_scale(model_path)

    print(f'[INFO] Model name: {model_name}')
    print(f'[INFO] Model scale: {model_scale}')

    super_res = cv2.dnn_superres.DnnSuperResImpl_create()
    super_res.readModel(model_path)
    super_res.setModel(model_name, model_scale)

    image = cv2.imread(image_path)
    print(f'[INFO] Original Scale - w: {image.shape[1]}, h: {image.shape[0]}')

    start = time.time()
    upscaled = super_res.upsample(image)
    end = time.time()
    print(f'[INFO] Super Resolution took {end-start:.6f} seconds')

    print(f'[INFO] Final Scale - w: {upscaled.shape[1]}, h: {upscaled.shape[0]}')

    start = time.time()
    bicubic = cv2.resize(
        image, (upscaled.shape[1], upscaled.shape[0]), interpolation=cv2.INTER_CUBIC
    )
    end = time.time()
    print(f'[INFO] Bicubic interpolation took {end-start:.6f} seconds')

    cv2.imshow('Original', image)
    cv2.imshow('Bicubic', bicubic)
    cv2.imshow('Super Resolution', upscaled)
    cv2.waitKey(0)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    default_model = os.path.join(curr_dir, 'models', 'ESPCN_x4.pb')
    default_image = os.path.join(curr_dir, '..', 'images', 'link.png')
    ap.add_argument('-m', '--model', help='path to model', default=default_model)
    ap.add_argument('-i', '--image', help='path to input image', default=default_image)
    args = vars(ap.parse_args())
    main(args['model'], args['image'])
