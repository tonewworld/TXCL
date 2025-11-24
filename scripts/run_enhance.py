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
