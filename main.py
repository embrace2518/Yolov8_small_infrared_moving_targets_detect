from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLOv8")
    parser.add_argument("--model", type=str, default="models/yolov8.pt")
    parser.add_argument("--source", type=str, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from validation import model_validate
    model_validate(args.model, args.source)

if __name__ == "__main__":
    main()
