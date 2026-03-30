from __future__ import annotations
import argparse
import os
from pathlib import Path
import yaml
from trainer import CustomTrainer, TrainingConfig

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="train the model")
    parser.add_argument("--config", type=str, default="train_config.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--debug-cuda", action="store_true", help="set CUDA_LAUNCH_BLOCKING=1 for accurate stack traces")
    parser.add_argument("--safe", action="store_true", help="force conservative training args for unstable CUDA environments")
    return parser.parse_args()

def load_training_config(config_path: str, args: argparse.Namespace) -> TrainingConfig:
    with Path(config_path).open("r", encoding="utf-8") as f:
        config_dict = yaml.safe_load(f) or {}
    if args.epochs is not None:
        config_dict["epochs"] = args.epochs
    if args.device is not None:
        config_dict["device"] = args.device
    if args.safe:
        config_dict["use_amp"] = False
        config_dict["half"] = False
        config_dict["cache"] = False
        config_dict["workers"] = 0
        config_dict["augment"] = False
        config_dict["multi_scale"] = False
        config_dict["batch_size"] = min(int(config_dict.get("batch_size", 8)), 4)
        config_dict["imgsz"] = min(int(config_dict.get("imgsz", 640)), 512)
    return TrainingConfig(**config_dict)

def main() -> None:
    args = parse_args()
    if args.debug_cuda:
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    config = load_training_config(args.config, args)
    trainer = CustomTrainer(config)
    trainer.train(resume_from=Path(args.resume) if args.resume else None)
    trainer.evaluate()

if __name__ == "__main__":
    main()
