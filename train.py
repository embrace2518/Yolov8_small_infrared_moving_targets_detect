
from __future__ import annotations
import argparse
from pathlib import Path
import yaml
from trainer import CustomTrainer, TrainingConfig

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="train the model")
    parser.add_argument("--config", type=str, default="train_config.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()

def load_training_config(config_path: str, args: argparse.Namespace) -> TrainingConfig:
    with Path(config_path).open("r", encoding="utf-8") as f:
        config_dict = yaml.safe_load(f) or {}
    if args.epochs is not None:
        config_dict["epochs"] = args.epochs
    if args.resume is not None:
        config_dict["resume"] = args.resume
    if args.device is not None:
        config_dict["device"] = args.device
    return TrainingConfig(**config_dict)

def main() -> None:
    args = parse_args()
    config = load_training_config(args.config, args)
    trainer = CustomTrainer(config)
    trainer.train(resume_from=Path(args.resume) if args.resume else None)
    trainer.evaluate()

if __name__ == "__main__":
    main()
