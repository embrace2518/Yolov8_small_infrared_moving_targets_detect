from __future__ import annotations
import argparse
from dataclasses import fields
from pathlib import Path
import yaml
from trainer import CustomTrainer, TrainingConfig

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="train the model")
    parser.add_argument("--config", type=str, default="train_config.yaml")
    parser.add_argument("--resume", type=str, default=None)
    return parser.parse_args()

def load_training_config(config_path: str) -> TrainingConfig:
    with Path(config_path).open("r", encoding="utf-8") as f:
        config_dict = yaml.safe_load(f) or {}

    # Backward-compat alias.
    if "preprocess_config_path" in config_dict and "pre_config_path" not in config_dict:
        config_dict["pre_config_path"] = config_dict.pop("preprocess_config_path")

    valid_keys = {f.name for f in fields(TrainingConfig)}
    filtered = {k: v for k, v in config_dict.items() if k in valid_keys}
    ignored = sorted(k for k in config_dict.keys() if k not in valid_keys)
    if ignored:
        print(f"[train.py] ignore unsupported config keys: {', '.join(ignored)}")

    return TrainingConfig(**filtered)

def plot_learning_curve(csv_path: Path | str, save_dir: Path | str):
    import pandas as pd
    import matplotlib.pyplot as plt
    try:
        df = pd.read_csv(csv_path)
        df.columns = df.columns.str.strip()
        plt.figure(figsize=(10, 5))
        if 'metrics/mAP50(B)' in df.columns:
            plt.plot(df['epoch'], df['metrics/mAP50(B)'], marker='o', label='mAP@50')
        if 'train/box_loss' in df.columns:
            plt.plot(df['epoch'], df['train/box_loss'], marker='x', label='Train Box Loss')
        plt.title('Training Learning Curve')
        plt.xlabel('Epoch')
        plt.ylabel('Score / Loss')
        plt.grid()
        plt.legend()
        out_file = Path(save_dir) / "custom_learning_curve.png"
        plt.savefig(str(out_file))
        plt.close()
        print(f"[Learning Curve] Saved to {out_file}")
    except Exception as e:
        print(f"[Learning Curve] Failed to plot: {e}")

def main() -> None:
    args = parse_args()
    config = load_training_config(args.config)
    trainer = CustomTrainer(config)
    trainer.train(resume_from=Path(args.resume) if args.resume else None)

    # Draw learning curve independently
    csv_path = trainer.output_dir / trainer.run_name / "results.csv"
    if csv_path.exists():
        plot_learning_curve(csv_path, trainer.output_dir / trainer.run_name)

    # == Run Custom Domain spotGEO Validation after default YOLOv8 Evaluation == #
    print("\n[train.py] Running ESA spotGEO custom validation on validation set...")
    val_source = config.val_data_dir[0]  # Just picking the first val folder
    best_weights_path = trainer.output_dir / trainer.run_name / "weights" / "best.pt"
    
    if best_weights_path.exists():
        from validation import model_validate
        model_validate(best_weights_path, str(val_source))

if __name__ == "__main__":
    main()
