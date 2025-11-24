"""Simple image quality/evaluation helpers.

Note: BRISQUE/NIQE are referenced in README. Installing their Python ports may require
additional packages (e.g. pybrisque or piq + pytorch). Here provide lightweight alternatives
for quick quantitative checks: global contrast (std), edge energy (Sobel variance).
"""
import numpy as np
import cv2


def contrast_std(img: np.ndarray) -> float:
    """Return standard deviation of luminance as a simple contrast proxy."""
    if img is None:
        return 0.0
    if img.ndim == 3:
        y = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        y = img
    return float(np.std(y))


def edge_energy(img: np.ndarray) -> float:
    """Sobel-based edge energy: higher usually means stronger edges/contours."""
    if img is None:
        return 0.0
    if img.ndim == 3:
        y = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        y = img
    sx = cv2.Sobel(y, cv2.CV_64F, 1, 0, ksize=3)
    sy = cv2.Sobel(y, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(sx * sx + sy * sy)
    return float(np.mean(mag))


# Placeholder functions for BRISQUE/NIQE. If desired, implement wrappers that call
# pybrisque or other packages and return scores.
def compute_brisque(img):
    raise NotImplementedError("BRISQUE not implemented. Install a BRISQUE package and implement a wrapper.")


def compute_niqe(img):
    raise NotImplementedError("NIQE not implemented. Install a NIQE package and implement a wrapper.")
