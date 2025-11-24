import os
import cv2
import numpy as np
from typing import Tuple


def read_image(path: str, as_gray: bool = False):
    if as_gray:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    else:
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def save_image(path: str, img: np.ndarray):
    # convert RGB back to BGR for cv2
    dirpath = os.path.dirname(path)
    if dirpath and not os.path.exists(dirpath):
        os.makedirs(dirpath, exist_ok=True)
    if img.ndim == 3 and img.shape[2] == 3:
        bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(path, bgr)
    else:
        cv2.imwrite(path, img)


def ensure_uint8(img: np.ndarray) -> np.ndarray:
    if img.dtype == np.uint8:
        return img
    img = np.clip(img, 0, 255)
    return img.astype(np.uint8)
