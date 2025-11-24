"""Enhancement routines: HE, CLAHE, contrast stretch, denoise, sharpen"""
import cv2
import numpy as np
from skimage import exposure


def hist_equalization(img: np.ndarray) -> np.ndarray:
    """Global histogram equalization. Works for color images by equalizing the Y channel."""
    if img is None:
        return None
    if img.ndim == 2:
        return cv2.equalizeHist(img)
    # color image: convert to YCrCb
    ycrcb = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    y_eq = cv2.equalizeHist(y)
    ycrcb_eq = cv2.merge([y_eq, cr, cb])
    return cv2.cvtColor(ycrcb_eq, cv2.COLOR_YCrCb2RGB)


def clahe_enhance(img: np.ndarray, clip_limit: float = 2.0, tile_grid_size=(8, 8)) -> np.ndarray:
    """CLAHE applied to the luminance channel for color images."""
    if img is None:
        return None
    if img.ndim == 2:
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        return clahe.apply(img)
    ycrcb = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    y_clahe = clahe.apply(y)
    ycrcb_clahe = cv2.merge([y_clahe, cr, cb])
    return cv2.cvtColor(ycrcb_clahe, cv2.COLOR_YCrCb2RGB)


def contrast_stretch(img: np.ndarray, low_perc: float = 2, high_perc: float = 98) -> np.ndarray:
    """Linear contrast stretching using percentile clipping. Works on color by channel."""
    if img is None:
        return None
    img_out = img.copy().astype(np.float32)
    if img.ndim == 2:
        p_low, p_high = np.percentile(img_out, (low_perc, high_perc))
        out = exposure.rescale_intensity(img_out, in_range=(p_low, p_high), out_range=(0, 255))
        return out.astype(np.uint8)
    for c in range(3):
        p_low, p_high = np.percentile(img_out[:, :, c], (low_perc, high_perc))
        img_out[:, :, c] = exposure.rescale_intensity(img_out[:, :, c], in_range=(p_low, p_high), out_range=(0, 255))
    return img_out.astype(np.uint8)


def denoise(img: np.ndarray, method: str = "median") -> np.ndarray:
    """Simple denoising options."""
    if img is None:
        return None
    if method == "median":
        if img.ndim == 2:
            return cv2.medianBlur(img, 3)
        return cv2.medianBlur(img, 3)
    if method == "bilateral":
        return cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)
    if method == "nlmeans":
        if img.ndim == 3:
            return cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
        else:
            return cv2.fastNlMeansDenoising(img, None, 10, 7, 21)
    # default: gaussian
    return cv2.GaussianBlur(img, (3, 3), 0)


def sharpen_edges(img: np.ndarray, amount: float = 1.0) -> np.ndarray:
    """Unsharp mask style sharpening to emphasize edges."""
    if img is None:
        return None
    blurred = cv2.GaussianBlur(img, (0, 0), 3)
    sharpened = cv2.addWeighted(img, 1.0 + amount, blurred, -amount, 0)
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def enhance_pipeline(img: np.ndarray, method: str = "clahe", denoise_method: str = None, sharpen: bool = True, **kwargs) -> np.ndarray:
    """High-level pipeline combining methods. method in ['clahe','he','stretch']"""
    if img is None:
        return None
    out = img.copy()
    if denoise_method:
        out = denoise(out, method=denoise_method)
    if method == "clahe":
        out = clahe_enhance(out, clip_limit=kwargs.get("clip_limit", 2.0), tile_grid_size=kwargs.get("tile_grid_size", (8, 8)))
    elif method == "he":
        out = hist_equalization(out)
    elif method == "stretch":
        out = contrast_stretch(out, low_perc=kwargs.get("low_perc", 2), high_perc=kwargs.get("high_perc", 98))
    if sharpen:
        out = sharpen_edges(out, amount=kwargs.get("sharpen_amount", 0.8))
    return out
