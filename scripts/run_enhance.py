"""CLI entrypoint for batch enhancement.

Usage examples (PowerShell):

python scripts\run_enhance.py --input data\raw --output data\processed --method clahe
python scripts\run_enhance.py --input data\raw --output data\processed --auto
"""
import argparse
import os
from txcl_enhance.io import batch_process_folder, batch_process_auto


def build_parser():
    p = argparse.ArgumentParser(description="Run batch enhancement on a folder of images")
    p.add_argument("--input", required=True, help="Input folder with images")
    p.add_argument("--output", required=True, help="Output folder to write enhanced images")
    p.add_argument("--method", default="clahe", help="Enhancement method: clahe, he, stretch")
    p.add_argument("--auto", action="store_true", help="Auto-detect distortion per-image and select method")
    return p


def main():
    parser = build_parser()
    args = parser.parse_args()
    inp = args.input
    out = args.output
    if not os.path.isdir(inp):
        raise SystemExit(f"Input folder does not exist: {inp}")
    os.makedirs(out, exist_ok=True)
    if args.auto:
        written = batch_process_auto(inp, out)
    else:
        written = batch_process_folder(inp, out, method=args.method)
    print(f"Wrote {len(written)} files to {out}")


if __name__ == '__main__':
    main()
"""Batch runner for enhancement (simple CLI)."""
import argparse
from txcl_enhance.io import batch_process_folder


def main():
    p = argparse.ArgumentParser(description="Batch enhance paint defect images")
    p.add_argument("--input", required=True, help="Input folder with raw images")
    p.add_argument("--output", required=True, help="Output folder for processed images")
    p.add_argument("--method", default="clahe", choices=["clahe", "he", "stretch"], help="Enhancement method")
    p.add_argument("--denoise", default=None, choices=[None, "median", "bilateral", "nlmeans"], help="Denoise method")
    args = p.parse_args()

    written = batch_process_folder(args.input, args.output, method=args.method, denoise_method=args.denoise)
    print(f"Wrote {len(written)} files to {args.output}")


if __name__ == "__main__":
    main()
