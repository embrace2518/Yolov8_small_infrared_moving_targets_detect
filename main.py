from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLOv8")
    parser.add_argument("--model", type=str, default="models/yolov8.pt")
    parser.add_argument("--source", type=str, required=True)
    parser.add_argument("--conf", type=float, default=0.1, help="置信度阈值")
    parser.add_argument("--save", action="store_true", help="是否保存预测可视化")
    parser.add_argument("--project", type=str, default="runs/detect")
    parser.add_argument("--name", type=str, default="predict")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # We now call our external validation module instead of dataset.dataset
    from validation import run_validate_json

    run_validate_json(
        model_path=args.model,
        source=args.source,
        conf=args.conf,
        save=args.save,
        project=args.project,
        name=args.name,
    )


if __name__ == "__main__":
    main()
