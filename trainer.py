"""
Trainer wrapper: keep custom preprocessing DataLoader, but delegate optimization/training loop to Ultralytics YOLO.train.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from collections.abc import Sized
from typing import Any, Optional

import cv2
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from ultralytics import YOLO
from dataset.dataset import DatasetConfig, EnhancedYOLODataset, yolo_collate_fn_with_indices
from dataset.preprocess import PipelineConfig, load_config as load_preprocess_config


@dataclass
class TrainingConfig:
    # data
    train_data_dir: Path
    val_data_dir: Path
    output_dir: Path = Path("runs/detect")
    preprocess_config_path: Path = Path("dataset/preprocess_config.yaml")

    # model
    base_model: str = "yolo11n.pt"
    pretrained: bool = True
    pretrained_weights: Optional[str] = None
    num_classes: int = 1
    class_names: list[str] = field(default_factory=lambda: ["target"])

    # train args
    epochs: int = 100
    batch_size: int = 8
    imgsz: int = 640
    device: str = "cuda"
    optimizer: str = "AdamW"
    learning_rate: float = 0.001
    weight_decay: float = 0.0005
    warmup_epochs: float = 2.0
    workers: int = 0  # Windows + pagefile pressure: keep default safe
    cache: bool = False
    use_amp: bool = True
    half: bool = False
    multi_scale: bool = False  # avoid random zero-size in unstable envs

    # aug
    augment: bool = True
    mosaic: float = 0.0
    mixup: float = 0.0
    hsv_h: float = 0.015
    hsv_s: float = 0.7
    hsv_v: float = 0.4

    # control
    seed: int = 42
    patience: int = 30
    save_period: int = 10
    run_name: Optional[str] = None
    refresh_staging: bool = True

    def __post_init__(self) -> None:
        self.train_data_dir = Path(self.train_data_dir)
        self.val_data_dir = Path(self.val_data_dir)
        self.output_dir = Path(self.output_dir)
        self.preprocess_config_path = Path(self.preprocess_config_path)
        self.class_names = list(self.class_names) if self.class_names else ["target"]


class CustomTrainer:
    """Use EnhancedYOLODataset/DataLoader to preprocess frames, then call YOLO.train."""

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = self._resolve_device(config.device)
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)

        self.output_dir = config.output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.run_name = config.run_name or datetime.now().strftime("exp_%Y%m%d_%H%M%S")
        self.staging_root = self.output_dir / "staging" / self.run_name
        self.dataset_yaml_path = self.staging_root / "dataset.yaml"

        self.preprocess_config: PipelineConfig = load_preprocess_config(config.preprocess_config_path)
        self.train_loader, self.val_loader = self.create_dataloaders()

        self.model = YOLO(self._resolve_model_source())
        self.last_train_results: Any = None

        print(f"[Trainer] device={self.device}")
        print(f"[Trainer] model={self._resolve_model_source()}")
        train_size = len(self.train_loader.dataset) if isinstance(self.train_loader.dataset, Sized) else -1
        val_size = len(self.val_loader.dataset) if isinstance(self.val_loader.dataset, Sized) else -1
        print(f"[Trainer] train images={train_size}")
        print(f"[Trainer] val images={val_size}")

    def _resolve_device(self, requested: str) -> str:
        if requested.startswith("cuda") and not torch.cuda.is_available():
            print("[Trainer] CUDA unavailable, fallback to CPU")
            return "cpu"
        return requested

    def _resolve_model_source(self) -> str:
        if self.config.pretrained and self.config.pretrained_weights:
            return self.config.pretrained_weights
        return self.config.base_model

    def _resolve_images_labels_dir(self, root: Path) -> tuple[Path, Path]:
        images_dir = root / "images"
        labels_dir = root / "labels"
        if images_dir.exists() and labels_dir.exists():
            return images_dir, labels_dir
        raise FileNotFoundError(
            f"Expected '{root}/images' and '{root}/labels'. Please organize dataset in YOLO folder layout."
        )

    def create_dataloaders(self) -> tuple[DataLoader, DataLoader]:
        train_images_dir, train_labels_dir = self._resolve_images_labels_dir(self.config.train_data_dir)
        val_images_dir, val_labels_dir = self._resolve_images_labels_dir(self.config.val_data_dir)

        train_dataset = EnhancedYOLODataset(
            DatasetConfig(
                images_dir=train_images_dir,
                labels_dir=train_labels_dir,
                preprocess_config=self.preprocess_config,
                augment=self.config.augment,
                target_size=(self.config.imgsz, self.config.imgsz),
                cache_images=False,
            ),
            mode="train",
        )
        val_dataset = EnhancedYOLODataset(
            DatasetConfig(
                images_dir=val_images_dir,
                labels_dir=val_labels_dir,
                preprocess_config=self.preprocess_config,
                augment=False,
                target_size=(self.config.imgsz, self.config.imgsz),
                cache_images=False,
            ),
            mode="val",
        )

        common_kwargs = {
            "batch_size": self.config.batch_size,
            "num_workers": self.config.workers,
            "pin_memory": self.device.startswith("cuda"),
            "collate_fn": yolo_collate_fn_with_indices,
        }
        train_loader = DataLoader(train_dataset, shuffle=False, drop_last=False, **common_kwargs)
        val_loader = DataLoader(val_dataset, shuffle=False, drop_last=False, **common_kwargs)
        return train_loader, val_loader

    def _save_batch_as_yolo_samples(self, images: torch.Tensor, targets: torch.Tensor, split_root: Path, start_index: int) -> int:
        images_dir = split_root / "images"
        labels_dir = split_root / "labels"
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)

        batch_size = int(images.shape[0])
        total_written = 0

        for local_idx in range(batch_size):
            global_idx = start_index + local_idx
            image_name = f"{global_idx:08d}.jpg"
            label_name = f"{global_idx:08d}.txt"

            # tensor [1, H, W] in [0,1] -> uint8 -> BGR for YOLO compatibility
            img = images[local_idx].detach().cpu().numpy()
            gray = np.clip(img[0] * 255.0, 0, 255).astype(np.uint8)
            bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            cv2.imwrite(str(images_dir / image_name), bgr)

            if targets.numel() > 0:
                selected = targets[targets[:, 0] == local_idx]
            else:
                selected = torch.zeros((0, 6), dtype=torch.float32)

            with (labels_dir / label_name).open("w", encoding="utf-8") as f:
                for row in selected:
                    cls_id = int(row[1].item())
                    xc = float(row[2].item())
                    yc = float(row[3].item())
                    w = float(row[4].item())
                    h = float(row[5].item())
                    f.write(f"{cls_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")

            total_written += 1

        return total_written

    def stage_preprocessed_dataset(self) -> Path:
        if self.staging_root.exists() and self.config.refresh_staging:
            for child in self.staging_root.iterdir():
                if child.is_dir():
                    for sub in child.rglob("*"):
                        if sub.is_file():
                            sub.unlink()
                else:
                    child.unlink()

        train_root = self.staging_root / "train"
        val_root = self.staging_root / "val"
        train_root.mkdir(parents=True, exist_ok=True)
        val_root.mkdir(parents=True, exist_ok=True)

        train_count = 0
        for images, targets in self.train_loader:
            train_count += self._save_batch_as_yolo_samples(images, targets, train_root, train_count)

        val_count = 0
        for images, targets in self.val_loader:
            val_count += self._save_batch_as_yolo_samples(images, targets, val_root, val_count)

        names = self.config.class_names
        dataset_yaml = {
            "path": str(self.staging_root.resolve()).replace("\\", "/"),
            "train": "train/images",
            "val": "val/images",
            "names": names,
            "nc": len(names),
        }

        with self.dataset_yaml_path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(dataset_yaml, f, sort_keys=False, allow_unicode=False)

        print(f"[Trainer] staged train={train_count}, val={val_count}")
        print(f"[Trainer] staged yaml={self.dataset_yaml_path}")
        return self.dataset_yaml_path

    def train(self, resume_from: Optional[Path] = None) -> Any:
        dataset_yaml = self.stage_preprocessed_dataset()

        # resume path: load ckpt as model then resume=True
        model_for_train = self.model
        resume_flag: bool = False
        if resume_from:
            ckpt = Path(resume_from)
            if ckpt.exists():
                model_for_train = YOLO(str(ckpt))
                resume_flag = True
                print(f"[Trainer] resume from {ckpt}")
            else:
                print(f"[Trainer] resume checkpoint not found: {ckpt}, train from scratch/weights")

        self.last_train_results = model_for_train.train(
            data=str(dataset_yaml),
            epochs=self.config.epochs,
            imgsz=self.config.imgsz,
            batch=self.config.batch_size,
            device=self.device,
            workers=self.config.workers,
            optimizer=self.config.optimizer,
            lr0=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            warmup_epochs=self.config.warmup_epochs,
            amp=self.config.use_amp,
            half=self.config.half and self.device.startswith("cuda"),
            augment=self.config.augment,
            multi_scale=self.config.multi_scale,
            mosaic=self.config.mosaic,
            mixup=self.config.mixup,
            hsv_h=self.config.hsv_h,
            hsv_s=self.config.hsv_s,
            hsv_v=self.config.hsv_v,
            cache=self.config.cache,
            seed=self.config.seed,
            patience=self.config.patience,
            save_period=self.config.save_period,
            project=str(self.output_dir),
            name=self.run_name,
            exist_ok=True,
            resume=resume_flag,
        )

        return self.last_train_results

    def evaluate(self) -> Any:
        if not self.dataset_yaml_path.exists():
            self.stage_preprocessed_dataset()
        return self.model.val(data=str(self.dataset_yaml_path), device=self.device)
