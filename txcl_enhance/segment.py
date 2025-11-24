"""Segmentation utilities for paint defect extraction.

Implements Bradley-Roth adaptive thresholding (integral image), morphological
post-processing (opening, reconstruction, hole filling), connected-component
filtering, and IoU evaluation helpers.
"""
import numpy as np
import cv2
from skimage import morphology
from skimage.morphology import reconstruction
from scipy import ndimage as ndi
from typing import Tuple


def bradley_threshold_gray(img: np.ndarray, win_size: int = 51, t: float = 0.15, mode: str = 'lower') -> np.ndarray:
    """Bradley-Roth adaptive threshold for a grayscale image.

    This implementation uses a fast local average computed by `cv2.boxFilter`,
    which internally uses summed-area tables (integral images). The threshold
    decision follows Bradley & Roth: compare pixel to local mean scaled by
    (1 +/- t).

    Args:
        img: uint8 grayscale image.
        win_size: window size (odd recommended).
        t: threshold constant (0..1).
        mode: 'lower' (dark defects), 'higher' (bright defects), or 'both'.

    Returns:
        binary mask (uint8 0/255)
    """
    if img is None:
        return None
    if img.ndim != 2:
        raise ValueError("bradley_threshold_gray expects a grayscale image")

    img_f = img.astype(np.float32)
    # local mean (box filter) — equivalent to integral-image average
    avg = cv2.boxFilter(img_f, ddepth=-1, ksize=(win_size, win_size), normalize=True, borderType=cv2.BORDER_REPLICATE)

    if mode == 'lower':
        mask = img_f < (avg * (1.0 - t))
    elif mode == 'higher':
        mask = img_f > (avg * (1.0 + t))
    else:
        mask = (img_f < (avg * (1.0 - t))) | (img_f > (avg * (1.0 + t)))

    return (mask.astype(np.uint8) * 255)


def morphological_reconstruction_cleanup(binary: np.ndarray, min_size: int = 50, open_radius: int = 3) -> np.ndarray:
    """Clean up binary mask using opening + morphological reconstruction, then
    fill holes and remove small objects.

    This follows the practice of using opening to remove small bright spots and
    then reconstructing to restore the shape of larger objects.
    """
    if binary is None:
        return None
    if binary.dtype != bool:
        bw = binary > 0
    else:
        bw = binary.copy()

    selem = morphology.disk(open_radius)
    opened = morphology.opening(bw, selem)

    # morphological reconstruction: dilate 'opened' under mask 'bw'
    # reconstruction requires float or uint8; use uint8
    seed = opened.astype(np.uint8)
    mask = bw.astype(np.uint8)
    # reconstruction by dilation: restores shapes of opened components constrained by mask
    rec = reconstruction(seed, mask, method='dilation')

    # fill holes
    filled = ndi.binary_fill_holes(rec > 0)

    # remove small objects
    cleaned = morphology.remove_small_objects(filled, min_size=min_size)

    return cleaned.astype(bool)


def largest_components_mask(bw: np.ndarray, keep_n: int = 1) -> np.ndarray:
    """Keep the largest `keep_n` connected components from boolean mask.

    Returns boolean mask.
    """
    if bw is None:
        return None
    lab, n = ndi.label(bw)
    if n == 0:
        return np.zeros_like(bw, dtype=bool)
    sizes = ndi.sum(np.ones_like(lab), lab, range(1, n + 1))
    # sizes length n
    idx = np.argsort(sizes)[::-1][:keep_n]
    keep_labels = [i + 1 for i in idx]
    out = np.isin(lab, keep_labels)
    return out


def compute_iou(pred: np.ndarray, gt: np.ndarray) -> float:
    """Compute IoU between two boolean or binary masks.

    Returns IoU in [0,1]. If both empty, returns 1.0.
    """
    if pred is None or gt is None:
        return 0.0
    p = pred > 0
    g = gt > 0
    inter = np.logical_and(p, g).sum()
    union = np.logical_or(p, g).sum()
    if union == 0:
        return 1.0 if inter == 0 else 0.0
    return float(inter) / float(union)


def segment_defects(img: np.ndarray, win_size: int = 41, t: float = 0.1, mode: str = 'both', min_size: int = 50, open_radius: int = 3, keep_n: int = 3) -> np.ndarray:
    """High-level pipeline: take RGB image, return binary mask of defects.

    Steps:
      - convert to grayscale
      - apply Bradley threshold
      - morphological cleanup
      - keep largest connected components
    """
    if img is None:
        return None
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img

    bt = bradley_threshold_gray(gray, win_size=win_size, t=t, mode=mode)
    clean = morphological_reconstruction_cleanup(bt, min_size=min_size, open_radius=open_radius)
    largest = largest_components_mask(clean, keep_n=keep_n)
    return largest.astype(np.uint8) * 255
