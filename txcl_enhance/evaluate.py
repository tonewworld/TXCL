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


def detect_distortion(img: np.ndarray, contrast_thresh: float = 10.0, blur_thresh: float = 100.0, noise_thresh: float = 10.0) -> dict:
    """Simple heuristic detector for common degradations.

    Returns a dict with keys:
      - 'contrast_std': float
      - 'edge_energy': float
      - 'lap_var': float  # variance of Laplacian (blurriness proxy)
      - 'noise_std': float
      - 'low_contrast': bool
      - 'blurry': bool
      - 'noisy': bool
      - 'suggested': dict with keys 'method', 'denoise_method', 'sharpen'

    The thresholds are heuristic and can be tuned per dataset.
    """
    if img is None:
        return {}
    # luminance
    if img.ndim == 3:
        y = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        y = img

    cstd = float(np.std(y))
    eeng = edge_energy(img)
    lap = cv2.Laplacian(y, cv2.CV_64F)
    lap_var = float(np.var(lap))

    # crude noise estimate: std of high-frequency residual (img - gaussian)
    blurred = cv2.GaussianBlur(y, (3, 3), 0)
    resid = y.astype(np.float32) - blurred.astype(np.float32)
    noise_std = float(np.std(resid))

    low_contrast = cstd < contrast_thresh
    blurry = lap_var < blur_thresh
    noisy = noise_std > noise_thresh

    # suggested mapping (simple heuristic)
    suggested = {"method": "clahe", "denoise_method": None, "sharpen": False}
    if low_contrast:
        # contrast enhancement (CLAHE preferred for local defects)
        suggested["method"] = "clahe"
    if noisy and not blurry:
        suggested["denoise_method"] = "nlmeans"
        # after denoising, do contrast stretch or clahe depending on contrast
        suggested["method"] = "stretch" if not low_contrast else "clahe"
    if blurry and not noisy:
        # primarily blur -> try sharpening after a conservative stretch
        suggested["method"] = "stretch"
        suggested["sharpen"] = True
    if blurry and noisy:
        # both: denoise then stretch + sharpen
        suggested["denoise_method"] = "nlmeans"
        suggested["method"] = "stretch"
        suggested["sharpen"] = True

    return {
        "contrast_std": cstd,
        "edge_energy": eeng,
        "lap_var": lap_var,
        "noise_std": noise_std,
        "low_contrast": low_contrast,
        "blurry": blurry,
        "noisy": noisy,
        "suggested": suggested,
    }
