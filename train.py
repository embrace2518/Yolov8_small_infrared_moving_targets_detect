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

def main() -> None:
    args = parse_args()
    config = load_training_config(args.config)
    trainer = CustomTrainer(config)
    trainer.train(resume_from=Path(args.resume) if args.resume else None)
    trainer.evaluate()

    # == Run Custom Domain spotGEO Validation after default YOLOv8 Evaluation == #
    print("\n[train.py] Running ESA spotGEO custom validation on validation set...")
    val_source = config.val_data_dir[0]  # Just picking the first val folder
    best_weights_path = trainer.output_dir / trainer.run_name / "weights" / "best.pt"
    
    if best_weights_path.exists():
        from validation import run_validate_json, compute_score
        out_json_path = trainer.output_dir / f"{trainer.run_name}_val" / "val_predictions.json"
        
        # 1. 运行预测生成JSON
        run_validate_json(
            model_path=best_weights_path,
            source=val_source,
            output_json=out_json_path,
            conf=0.05,        # 稍低阈值获取召回率
            project=trainer.output_dir,
            name=f"{trainer.run_name}_val_custom"
        )
        
        print(f"[train.py] Custom prediction output saved to {out_json_path}")
        
        # 2. 如果存在真实标签 true_labels_json_path，自动进行赛道得分评测
        true_labels_json = config.pre_config_path.parent / "true_labels.json"
        if true_labels_json.exists():
            score, mse = compute_score(str(out_json_path), str(true_labels_json))
            print(f"\n======================================")
            print(f"🌟 END OF TRAINING: spotGEO COMPETITION SCORE 🌟")
            print(f"Score (1 - F1): {score:0.6f}")
            print(f"MSE           : {mse:0.6f}")
            print(f"======================================\n")
        else:
            print(f"[train.py] Note: Could not find '{true_labels_json}', skipping exact score computation.")
    else:
        print("[train.py] Warning: best.pt not found, skipping custom validation.")

if __name__ == "__main__":
    main()
