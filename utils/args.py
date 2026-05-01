"""Shared argparse helpers — keeps argument definitions DRY across entry points."""
from __future__ import annotations

import argparse


def add_common_eval_args(parser: argparse.ArgumentParser) -> None:
    """Add evaluation/prediction arguments shared by main.py and evaluate.py."""
    parser.add_argument(
        "--weights", type=str, default=None,
        help="模型权重路径 (.pt)",
    )
    parser.add_argument(
        "--conf", type=float, default=0.1,
        help="检测置信度阈值（默认 0.1）",
    )
    parser.add_argument(
        "--preprocess", action="store_true",
        help="启用预处理（NUC+去噪+CLAHE+Gamma）",
    )
    parser.add_argument(
        "--output-dir", type=str, default="runs/evaluate",
        help="输出目录",
    )
    parser.add_argument(
        "--batch-size", type=int, default=32,
        help="推理批量大小（默认 32）",
    )
    parser.add_argument(
        "--img-size", type=int, default=640,
        help="图像输入尺寸（默认 640）",
    )
