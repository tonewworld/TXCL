import numpy as np
from txcl_enhance.enhance import hist_equalization, clahe_enhance, contrast_stretch, enhance_pipeline


def make_synthetic():
    # create a dark image with a brighter circular defect
    img = np.zeros((128, 128, 3), dtype=np.uint8)
    cv2 = None
    try:
        import cv2 as _cv
        cv2 = _cv
    except Exception:
        cv2 = None
    if cv2:
        cv2.circle(img, (64, 64), 20, (120, 120, 120), -1)
    else:
        # fallback: draw with numpy
        rr, cc = np.ogrid[:128, :128]
        mask = (rr - 64) ** 2 + (cc - 64) ** 2 <= 20 ** 2
        img[mask] = 120
    return img


def test_enhance_pipeline_basic():
    img = make_synthetic()
    out = enhance_pipeline(img, method="clahe", denoise_method=None, sharpen=False)
    assert out is not None
    assert out.shape == img.shape
