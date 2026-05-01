"""Quick inference entry point. For full evaluation with metrics, use evaluate.py."""
from __future__ import annotations

import argparse
from pathlib import Path

from evaluate import evaluate
from utils.args import add_common_eval_args


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Quick inference with YOLOv8 model on a directory of images."
    )
    add_common_eval_args(parser)
    # Override: --weights is required for main.py
    for action in parser._actions:
        if action.dest == "weights":
            action.required = True
            break
    parser.add_argument(
        "--source", type=str, required=True,
        help="Path to image directory or file"
    )
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
