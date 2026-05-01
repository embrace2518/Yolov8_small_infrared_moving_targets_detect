from __future__ import annotations

import argparse

from evaluate import evaluate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLOv8")
    parser.add_argument("--model", type=str, default="models/yolov8.pt")
    parser.add_argument("--source", type=str, required=True)
    return parser.parse_args()


def main() -> None:
    # args = parse_args()
    # evaluate(args.model, args.source)

    evaluate(
        weights='runs/detect/exp_20260429_094305/weights/best.pt',
        sources='D:/Dataset/test/1',
    )


if __name__ == "__main__":
    main()
