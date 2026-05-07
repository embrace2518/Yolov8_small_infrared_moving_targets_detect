from __future__ import annotations
import argparse
from dataclasses import fields
from pathlib import Path
import yaml
from ultralytics.utils.plotting import plot_results
from trainer import TrainingConfig, CustomTrainer
from utils.logging import get_logger

logger = get_logger(__name__)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="train the model")
    parser.add_argument("--config", type=str, default="train_config.yaml")
    parser.add_argument("--resume", type=str, default=None)
    return parser.parse_args()

def load_training_config(config_path: str) -> TrainingConfig:
    with Path(config_path).open("r", encoding="utf-8") as f:
        config_dict = yaml.safe_load(f) or {}

    # Backward-compat aliases.
    if "preprocess_config_path" in config_dict and "pre_config_path" not in config_dict:
        config_dict["pre_config_path"] = config_dict.pop("preprocess_config_path")
    if "dataset_config_path" not in config_dict and "pre_config_path" in config_dict:
        config_dict["dataset_config_path"] = config_dict["pre_config_path"]

    valid_keys = {f.name for f in fields(TrainingConfig)}
    filtered = {k: v for k, v in config_dict.items() if k in valid_keys}
    ignored = sorted(k for k in config_dict.keys() if k not in valid_keys)
    if ignored:
        logger.warning("ignore unsupported config keys: %s", ", ".join(ignored))

    return TrainingConfig(**filtered)

def main() -> None:
    args = parse_args()
    config = load_training_config(args.config)
    trainer = CustomTrainer(config)
    trainer.train(resume_from=Path(args.resume) if args.resume else None)

    # Use YOLO's actual save_dir (may differ from self-constructed path due to exist_ok reuse)
    save_dir = trainer.model.trainer.save_dir
    csv_path = save_dir / "results.csv"
    if csv_path.exists():
        plot_results(csv_path, save_dir)
    else:
        logger.warning("results.csv not found at %s, skipping plot_results", csv_path)

    # == Run Custom Domain spotGEO Validation after default YOLOv8 Evaluation == #
    logger.info("Running ESA spotGEO custom validation on validation set...")
    val_sources = config.val_data_dir
    best_weights_path = save_dir / "weights" / "best.pt"
    
    if best_weights_path.exists():
        from evaluate import evaluate
        evaluate(weights=str(best_weights_path), sources=val_sources, no_save=True)

if __name__ == "__main__":
    main()
