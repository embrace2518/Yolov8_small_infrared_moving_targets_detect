from __future__ import annotations

import argparse

from validation import model_validate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLOv8")
    parser.add_argument("--model", type=str, default="models/yolov8.pt")
    parser.add_argument("--source", type=str, required=True)
    return parser.parse_args()


def main() -> None:
    # args = parse_args()
    # from validation import model_validate
    # model_validate(args.model, args.source)

    from ultralytics.utils.plotting import plot_results

    # 绘制训练结果
    # plot_results(file='runs/detect/exp_20260429_094305/results.csv', dir='.')
    model_validate('runs/detect/exp_20260429_094305/weights/best.pt', 'D:/Dataset/test/1')
if __name__ == "__main__":
    main()
