import os
from typing import List
from .utils import read_image, save_image
from .enhance import enhance_pipeline
from .evaluate import detect_distortion


def batch_process_folder(input_dir: str, output_dir: str, method: str = "clahe", **kwargs) -> List[str]:
    """Process all image files in input_dir and write to output_dir. Returns list of written files."""
    written = []
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    for fname in os.listdir(input_dir):
        if not fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")):
            continue
        in_path = os.path.join(input_dir, fname)
        out_path = os.path.join(output_dir, fname)
        img = read_image(in_path, as_gray=False)
        if img is None:
            continue
        out = enhance_pipeline(img, method=method, **kwargs)
        save_image(out_path, out)
        written.append(out_path)
    return written


def batch_process_auto(input_dir: str, output_dir: str, **kwargs) -> List[str]:
    """Process images and auto-detect distortion per-image to pick enhancement strategy."""
    written = []
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    for fname in os.listdir(input_dir):
        if not fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")):
            continue
        in_path = os.path.join(input_dir, fname)
        out_path = os.path.join(output_dir, fname)
        img = read_image(in_path, as_gray=False)
        if img is None:
            continue
        det = detect_distortion(img)
        sugg = det.get("suggested", {})
        out = enhance_pipeline(img, method=sugg.get("method", "clahe"), denoise_method=sugg.get("denoise_method", None), sharpen=sugg.get("sharpen", True), **kwargs)
        save_image(out_path, out)
        written.append(out_path)
    return written
