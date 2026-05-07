"""Quick inference entry point. For full evaluation with metrics, use evaluate.py."""
from __future__ import annotations

import argparse
from pathlib import Path

from evaluate import evaluate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Quick inference with YOLOv8 model on a directory of images."
    )
    parser.add_argument("--weights", type=str, required=True, help="模型权重路径 (.pt)")
    parser.add_argument("--source", type=str, required=True, help="图像目录或文件路径")
    parser.add_argument("--conf", type=float, default=0.1, help="检测置信度阈值（默认 0.1）")
    parser.add_argument("--preprocess", action="store_true", help="启用预处理（NUC+去噪+CLAHE+Gamma）")
    parser.add_argument("--output-dir", type=str, default="runs/evaluate", help="输出目录")
    parser.add_argument("--batch-size", type=int, default=32, help="推理批量大小（默认 32）")
    parser.add_argument("--img-size", type=int, default=640, help="图像输入尺寸（默认 640）")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not Path(args.weights).exists():
        raise FileNotFoundError(f"Weights file not found: {args.weights}")
    if not Path(args.source).exists():
        raise FileNotFoundError(f"Source not found: {args.source}")

    evaluate(
        weights=args.weights,
        sources=args.source,
        enable_preprocess=args.preprocess,
        conf=args.conf,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
