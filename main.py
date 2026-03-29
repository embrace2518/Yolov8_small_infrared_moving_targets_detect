from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO
from dataset.convert_json import model_validate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLOv8 inference entry")
    parser.add_argument("--model", type=str, default="models/yolov8.pt", help="权重路径")
    parser.add_argument("--source", type=str, required=True, help="图片/目录/视频路径")
    parser.add_argument("--conf", type=float, default=0.1, help="置信度阈值")
    parser.add_argument("--save", action="store_true", help="是否保存预测可视化")
    parser.add_argument("--project", type=str, default="runs/detect", help="保存目录")
    parser.add_argument("--name", type=str, default="predict", help="实验名")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = YOLO(args.model)
    predictions = model.predict(
        source=str(Path(args.source)),
        conf=args.conf,
        save=args.save,
        project=args.project,
        name=args.name,
        exist_ok=True,
    )
    model_validate(predictions)


if __name__ == "__main__":
    main()
