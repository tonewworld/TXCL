"""Batch enhance images and produce evaluation metrics.

Saves enhanced images to the output folder and writes a CSV with per-image
metrics (contrast_std, edge_energy) before and after enhancement.

Example (PowerShell):
python scripts\run_and_evaluate.py --input data\raw\images --output data\enhance --method clahe
python scripts\run_and_evaluate.py --input data\raw\images --output data\enhance --auto
"""
import os
import argparse
import csv
from typing import List

# Ensure project root is on sys.path when running the script directly from the
# `scripts/` folder (e.g. `python scripts\run_and_evaluate.py`). Without this,
# Python may not find the local package `txcl_enhance` because sys.path[0]
# becomes the `scripts/` directory.
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from txcl_enhance.utils import read_image, save_image, ensure_uint8
from txcl_enhance.enhance import enhance_pipeline
from txcl_enhance.evaluate import contrast_std, edge_energy, detect_distortion


def gather_image_files(input_dir: str) -> List[str]:
    imgs = []
    if not os.path.isdir(input_dir):
        return imgs
    for fname in sorted(os.listdir(input_dir)):
        if fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")):
            imgs.append(os.path.join(input_dir, fname))
    return imgs


def process_all(input_dir: str, output_dir: str, csv_path: str, method: str = "clahe", denoise: str = None, auto: bool = False, **kwargs):
    os.makedirs(output_dir, exist_ok=True)
    imgs = gather_image_files(input_dir)
    if not imgs:
        print(f"No images found in {input_dir}")
        return

    rows = []
    for p in imgs:
        print(f"Processing {os.path.basename(p)}...")
        img = read_image(p, as_gray=False)
        if img is None:
            print(f"  Failed to read {p}, skipping")
            continue
        before_c = contrast_std(img)
        before_e = edge_energy(img)

        use_method = method
        use_denoise = denoise
        use_sharpen = True
        if auto:
            det = detect_distortion(img)
            sugg = det.get("suggested", {})
            use_method = sugg.get("method", method)
            use_denoise = sugg.get("denoise_method", denoise)
            use_sharpen = sugg.get("sharpen", True)

        out = enhance_pipeline(img, method=use_method, denoise_method=use_denoise, sharpen=use_sharpen, **kwargs)
        out = ensure_uint8(out)
        out_name = os.path.basename(p)
        out_path = os.path.join(output_dir, out_name)
        save_image(out_path, out)

        after_c = contrast_std(out)
        after_e = edge_energy(out)

        rows.append({
            "file": out_name,
            "in_path": p,
            "out_path": out_path,
            "method": use_method,
            "denoise": str(use_denoise),
            "before_contrast": before_c,
            "after_contrast": after_c,
            "before_edge_energy": before_e,
            "after_edge_energy": after_e,
        })

    # write CSV
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["file", "in_path", "out_path", "method", "denoise", "before_contrast", "after_contrast", "before_edge_energy", "after_edge_energy"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    # summary
    def mean(xs):
        return sum(xs) / len(xs) if xs else 0.0

    before_cs = [r["before_contrast"] for r in rows]
    after_cs = [r["after_contrast"] for r in rows]
    before_es = [r["before_edge_energy"] for r in rows]
    after_es = [r["after_edge_energy"] for r in rows]

    print("\nEvaluation summary:")
    print(f"  Images processed: {len(rows)}")
    print(f"  Contrast (std)  - before: {mean(before_cs):.3f}, after: {mean(after_cs):.3f}, delta: {mean(after_cs)-mean(before_cs):.3f}")
    print(f"  Edge energy     - before: {mean(before_es):.3f}, after: {mean(after_es):.3f}, delta: {mean(after_es)-mean(before_es):.3f}")
    print(f"  CSV saved to: {csv_path}")


def build_parser():
    p = argparse.ArgumentParser(description="Batch enhance and evaluate images")
    p.add_argument("--input", required=False, default="data/raw/images", help="Input folder")
    p.add_argument("--output", required=False, default="data/enhance", help="Output folder for enhanced images")
    p.add_argument("--csv", required=False, default="data/enhance/metrics.csv", help="CSV output path")
    p.add_argument("--method", default="clahe", choices=["clahe", "he", "stretch", "gamma"], help="Enhancement method")
    p.add_argument("--denoise", default=None, choices=[None, "median", "bilateral", "nlmeans"], help="Denoise method")
    p.add_argument("--auto", action="store_true", help="Auto-detect distortion and select method per-image")
    p.add_argument("--clip_limit", type=float, default=2.0, help="CLAHE clip limit")
    p.add_argument("--tile_grid", type=int, nargs=2, default=(16, 16), help="CLAHE tile grid size (two ints)")
    p.add_argument("--low_perc", type=float, default=2.0, help="Contrast stretch low percentile")
    p.add_argument("--high_perc", type=float, default=98.0, help="Contrast stretch high percentile")
    p.add_argument("--gamma", type=float, default=1.2, help="Gamma correction value")
    p.add_argument("--sharpen_amount", type=float, default=0.5, help="Sharpen amount")
    return p


def main():
    parser = build_parser()
    args = parser.parse_args()
    tile = tuple(args.tile_grid)
    kwargs = {
        "clip_limit": args.clip_limit,
        "tile_grid_size": tile,
        "low_perc": args.low_perc,
        "high_perc": args.high_perc,
        "gamma": args.gamma,
        "sharpen_amount": args.sharpen_amount,
    }
    csv_path = args.csv
    # ensure csv dir exists
    csv_dir = os.path.dirname(csv_path)
    if csv_dir:
        os.makedirs(csv_dir, exist_ok=True)

    process_all(args.input, args.output, csv_path, method=args.method, denoise=args.denoise, auto=args.auto, **kwargs)


if __name__ == "__main__":
    main()
