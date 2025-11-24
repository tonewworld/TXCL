"""Batch segmentation of defects from enhanced or raw images.

Saves binary masks and optional overlay visualization. If a ground-truth mask
folder is provided (same basenames), computes IoU per-image and prints a
summary.

Example:
python scripts\run_segment.py --input data\raw\images --output data\segment --gt data\raw\labels
python scripts\run_segment.py --input data\raw\images --output data\segment --enhance --method clahe
"""
import os
import argparse
import sys
import numpy as np
from typing import List

import os
import argparse
import sys
from typing import List
import numpy as np

# ensure project root is importable when running from scripts/
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from txcl_enhance.utils import read_image, save_image, ensure_uint8
from txcl_enhance.enhance import enhance_pipeline
from txcl_enhance.segment import segment_defects, compute_iou


def gather_image_files(input_dir: str) -> List[str]:
    imgs = []
    if not os.path.isdir(input_dir):
        return imgs
    for fname in sorted(os.listdir(input_dir)):
        if fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")):
            imgs.append(os.path.join(input_dir, fname))
    return imgs


def overlay_mask(img, mask, alpha=0.4):
    # img: RGB uint8, mask: binary 0/255
    if img is None:
        return None
    vis = img.copy()
    red = np.zeros_like(vis)
    red[:, :, 0] = 255
    m = (mask > 0)
    vis[m] = (vis[m] * (1 - alpha) + red[m] * alpha).astype('uint8')
    return vis


def process(input_dir: str, output_dir: str, gt_dir: str = None, enhance: bool = False, method: str = 'clahe', **kwargs):
    os.makedirs(output_dir, exist_ok=True)
    mask_dir = os.path.join(output_dir, 'masks')
    vis_dir = os.path.join(output_dir, 'vis')
    os.makedirs(mask_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)

    imgs = gather_image_files(input_dir)
    if not imgs:
        print('No images found')
        return

    ious = []
    for p in imgs:
        name = os.path.basename(p)
        print('Processing', name)
        img = read_image(p, as_gray=False)
        if img is None:
            print('  failed to read, skipping')
            continue
        proc = img
        if enhance:
            proc = enhance_pipeline(img, method=method, **kwargs)
        mask = segment_defects(proc)
        mask_u8 = ensure_uint8(mask)
        mask_path = os.path.join(mask_dir, name.rsplit('.', 1)[0] + '.png')
        save_image(mask_path, mask_u8)
        # overlay visualization
        try:
            import numpy as np
            vis = overlay_mask(img, mask_u8)
            vis_path = os.path.join(vis_dir, name)
            save_image(vis_path, vis)
        except Exception:
            pass

        # compute IoU if ground truth mask exists
        if gt_dir:
            gt_candidate_png = os.path.join(gt_dir, name.rsplit('.', 1)[0] + '.png')
            gt_candidate_jpg = os.path.join(gt_dir, name)
            gt_mask = None
            if os.path.exists(gt_candidate_png):
                gt_mask = read_image(gt_candidate_png, as_gray=True)
            elif os.path.exists(gt_candidate_jpg):
                gt_mask = read_image(gt_candidate_jpg, as_gray=True)
            if gt_mask is not None:
                iou = compute_iou(mask_u8, gt_mask)
                ious.append(iou)
                print(f'  IoU: {iou:.3f}')

    if ious:
        import statistics
        print('\nIoU summary:')
        print(f'  count: {len(ious)}, mean: {statistics.mean(ious):.3f}, median: {statistics.median(ious):.3f}')


def build_parser():
    p = argparse.ArgumentParser(description='Run segmentation on folder of images')
    p.add_argument('--input', required=True, help='Input folder')
    p.add_argument('--output', required=True, help='Output folder')
    p.add_argument('--gt', required=False, default=None, help='Ground-truth mask folder (optional)')
    p.add_argument('--enhance', action='store_true', help='Apply enhancement before segmentation')
    p.add_argument('--method', default='clahe', choices=['clahe', 'he', 'stretch'], help='Enhancement method')
    p.add_argument('--win', type=int, default=51, help='Bradley window size')
    p.add_argument('--t', type=float, default=0.15, help='Bradley threshold constant')
    p.add_argument('--mode', default='both', choices=['lower', 'higher', 'both'], help='Threshold mode: lower (dark defects), higher (bright defects), or both')
    p.add_argument('--min_size', type=int, default=50, help='Minimum connected component size')
    p.add_argument('--open_radius', type=int, default=3, help='Opening radius')
    p.add_argument('--keep_n', type=int, default=3, help='Keep largest N components')
    return p


def main():
    parser = build_parser()
    args = parser.parse_args()
    process(args.input, args.output, gt_dir=args.gt, enhance=args.enhance, method=args.method, win_size=args.win, t=args.t, mode=args.mode, min_size=args.min_size, open_radius=args.open_radius, keep_n=args.keep_n)


if __name__ == '__main__':
    main()
