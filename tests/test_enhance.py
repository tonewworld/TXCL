import numpy as np
from txcl_enhance.enhance import hist_equalization, clahe_enhance, contrast_stretch, enhance_pipeline
from txcl_enhance.evaluate import detect_distortion
import cv2


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


def test_detect_low_contrast():
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    cv2.circle(img, (32, 32), 10, (120, 120, 120), -1)
    det = detect_distortion(img)
    assert det["low_contrast"] is True or det["contrast_std"] < 30


def test_detect_noisy_and_blur():
    base = np.full((64, 64), 128, dtype=np.uint8)
    noise = (np.random.randn(*base.shape) * 25).astype(np.int16)
    noisy = np.clip(base.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    noisy = cv2.cvtColor(noisy, cv2.COLOR_GRAY2RGB)
    detn = detect_distortion(noisy)
    assert detn["noisy"] is True

    blurred = cv2.GaussianBlur(noisy, (9, 9), 5)
    detb = detect_distortion(blurred)
    assert detb["blurry"] is True
