"""Lightweight CLI helpers (could be expanded)."""
import argparse


def build_parser():
    p = argparse.ArgumentParser(description="txcl-enhance helper CLI")
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--method", default="clahe")
    return p


def main():
    parser = build_parser()
    args = parser.parse_args()
    print(args)


if __name__ == '__main__':
    main()
