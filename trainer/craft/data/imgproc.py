"""  
Copyright (c) 2019-present NAVER Corp.
MIT License
"""

# -*- coding: utf-8 -*-
import numpy as np

import cv2
from skimage import io


def loadImage(img_file):
    img = io.imread(img_file)  # RGB order
    if img.shape[0] == 2:
        img = img[0]
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    if img.shape[2] == 4:
        img = img[:, :, :3]
    img = np.array(img)

    return img


def normalizeMeanVariance(
    in_img, mean=(0.485, 0.456, 0.406), variance=(0.229, 0.224, 0.225)
):
    # should be RGB order
    img = in_img.copy().astype(np.float32)

    img -= np.array(
        [mean[0] * 255.0, mean[1] * 255.0, mean[2] * 255.0], dtype=np.float32
    )
    img /= np.array(
        [variance[0] * 255.0, variance[1] * 255.0, variance[2] * 255.0],
        dtype=np.float32,
    )
    return img


def denormalizeMeanVariance(
    in_img, mean=(0.485, 0.456, 0.406), variance=(0.229, 0.224, 0.225)
):
    # should be RGB order
    img = in_img.copy()
    img *= variance
    img += mean
    img *= 255.0
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img


def resize_aspect_ratio(img, square_size, interpolation, mag_ratio=1):
    height, width, channel = img.shape

    target_size = (768, 768)

    # magnify image size
    # target_size = mag_ratio * max(height, width)

    # set original image size
    # if target_size > square_size:
    #     target_size = square_size

    ratio = (target_size[0]/height, target_size[1]/width)

    target_h, target_w = target_size[0], target_size[1]

    # print(f"printing from improc.py | target_h, target_width: {target_h, target_w}")

    # NOTE
    valid_size_heatmap = (int(target_h / 2), int(target_w / 2))

    # print(f"printing from improc.py | valid_size_heatmap: {valid_size_heatmap}")

    proc = cv2.resize(img, (target_w, target_h), interpolation=interpolation)

    # make canvas and paste image
    target_h32, target_w32 = target_h, target_w
    if target_h % 32 != 0:
        target_h32 = target_h + (32 - target_h % 32)
    if target_w % 32 != 0:
        target_w32 = target_w + (32 - target_w % 32)
    resized = np.zeros((target_h32, target_w32, channel), dtype=np.float32)
    resized[0:target_h, 0:target_w, :] = proc

    # target_h, target_w = target_h32, target_w32
    # size_heatmap = (int(target_w/2), int(target_h/2))

    # resized = cv2.resize(resized, (768, 768))

    # print(f"print from imgproc.py | resized: {resized.shape}")
    # print(f"print from imgproc.py | ratio: {ratio}")
    # print(f"print from imgproc.py | valid_size_heatmap: {valid_size_heatmap}")

    return resized, ratio, valid_size_heatmap


def cvt2HeatmapImg(img):
    img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    img = cv2.applyColorMap(img, cv2.COLORMAP_JET)
    return img
