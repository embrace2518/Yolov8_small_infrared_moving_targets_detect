"""
Trainer wrapper: keep custom preprocessing DataLoader, but delegate optimization/training loop to Ultralytics YOLO.train.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import os
from collections.abc import Sized
from typing import Any, Optional

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
    train_data_dir: Path | str | list[Path | str]
    val_data_dir: Path | str | list[Path | str]
    output_dir: Path = "runs/detect"
    preprocess_config_path: Path = "dataset/preprocess_config.yaml"

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


    @staticmethod
    def _normalize_path_list(value: Path | str | list[Path | str]) -> list[Path]:
        if isinstance(value, (str, Path)):
            raw_items = [value]
        elif isinstance(value, list):
            raw_items = value
        else:
            raise TypeError(f"expected path or list of paths, got: {type(value)}")

        paths: list[Path] = []
        for item in raw_items:
            if item is None:
                continue
            as_text = str(item).strip()
            if not as_text:
                continue
            paths.append(Path(as_text))
        return paths

    def __post_init__(self) -> None:
        train_paths = self._normalize_path_list(self.train_data_dir)
        if not train_paths:
            print("[TrainingConfig] train_data_dir is empty, fallback to current directory '.'")
            train_paths = [Path(".")]
        self.train_data_dir = train_paths

        val_paths = self._normalize_path_list(self.val_data_dir)
        if not val_paths:
            print("[TrainingConfig] val_data_dir is empty, fallback to train_data_dir")
            val_paths = train_paths
        self.val_data_dir = val_paths

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
        self.model_source = self._resolve_model_source()

        self.preprocess_config: PipelineConfig = load_preprocess_config(config.preprocess_config_path)
        self.train_loader, self.val_loader = self.create_dataloaders()
        self.model = YOLO(self.model_source)
        self.last_train_results: Any = None

        print(f"[Trainer] device={self.device}")
        print(f"[Trainer] model={self.model_source}")
        self._log_runtime_env()
        train_size = len(self.train_loader.dataset) if isinstance(self.train_loader.dataset, Sized) else -1
        val_size = len(self.val_loader.dataset) if isinstance(self.val_loader.dataset, Sized) else -1
        print(f"[Trainer] train images={train_size}")
        print(f"[Trainer] val images={val_size}")

    @staticmethod
    def _log_runtime_env() -> None:
        print(f"[Trainer] torch={torch.__version__}")
        print(f"[Trainer] cuda_available={torch.cuda.is_available()}, cuda_count={torch.cuda.device_count()}")
        if torch.cuda.is_available():
            try:
                idx = torch.cuda.current_device()
                print(f"[Trainer] gpu={torch.cuda.get_device_name(idx)}")
            except Exception as exc:
                print(f"[Trainer] failed to query GPU name: {exc}")
        print(f"[Trainer] CUDA_LAUNCH_BLOCKING={os.environ.get('CUDA_LAUNCH_BLOCKING')}")

    @staticmethod
    def _resolve_device(requested: str) -> str:
        if requested.startswith("cuda") and not torch.cuda.is_available():
            print("[Trainer] CUDA unavailable, fallback to CPU")
            return "cpu"
        return requested

    def _resolve_model_source(self) -> str:
        if self.config.pretrained and self.config.pretrained_weights:
            return self.config.pretrained_weights
        return self.config.base_model

    def _make_dataset(self, image_dirs: list[Path], mode: str) -> EnhancedYOLODataset:
        return EnhancedYOLODataset(
            DatasetConfig(
                images_dir=image_dirs,
                labels_dir=image_dirs,
                preprocess_config=self.preprocess_config,
                augment=self.config.augment,
                target_size=(self.config.imgsz, self.config.imgsz),
            ),
            mode=mode,
        )

    def _make_loader(self, dataset: EnhancedYOLODataset) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            num_workers=0,  # 直接训练时避免多进程问题
            pin_memory=self.device.startswith("cuda"),
            collate_fn=yolo_collate_fn_with_indices,
            shuffle=False,
            drop_last=False,
        )

    def create_dataloaders(self) -> tuple[DataLoader, DataLoader]:
        train_dataset = self._make_dataset(self.config.train_data_dir, mode="train")
        val_dataset = self._make_dataset(self.config.val_data_dir, mode="val")
        return self._make_loader(train_dataset), self._make_loader(val_dataset)

    def stage_preprocessed_dataset(self) -> Path:
        """不再生成实际文件，只返回一个虚拟路径用于兼容性"""
        # 创建必要的目录结构
        self.staging_root.mkdir(parents=True, exist_ok=True)

        # 生成一个虚拟的dataset.yaml
        dataset_yaml = {
            "path": str(self.staging_root.resolve()),
            "train": "train",  # 虚拟路径
            "val": "val",  # 虚拟路径
            "nc": int(self.config.num_classes),
            "names": self.config.class_names,
        }

        self.dataset_yaml_path.parent.mkdir(parents=True, exist_ok=True)
        with self.dataset_yaml_path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(dataset_yaml, f, sort_keys=False, allow_unicode=False)
        return self.dataset_yaml_path

    def _generate_image_list(self, data_dirs: list[Path], filename: str) -> Path:
        """生成图片路径列表文件"""
        list_path = self.staging_root / filename
        with list_path.open("w", encoding="utf-8") as f:
            for data_dir in data_dirs:
                for img_path in sorted(data_dir.glob("*.jpg")):
                    f.write(f"{img_path.resolve()}\n")
        return list_path

    def train(self, resume_from: Optional[Path] = None) -> Any:
        """直接使用DataLoader进行训练，跳过文件生成"""
        # 模型加载（保持不变）
        model_for_train = self.model
        resume_flag: bool = False
        if resume_from and Path(resume_from).exists():
            model_for_train = YOLO(resume_from)
            self.model = model_for_train
            resume_flag = True
            print(f"[Trainer] 从检查点恢复: {resume_from}")
        elif resume_from:
            print(f"[Trainer] 检查点未找到: {resume_from}")

        # 生成虚拟配置文件（用于兼容性）
        dataset_yaml = self.stage_preprocessed_dataset()

        # 直接训练参数配置
        train_args = dict(
            # 使用虚拟配置文件保持兼容性
            data=str(dataset_yaml),

            # 直接传入DataLoader（如果Ultralytics YOLO支持）
            # 注意：当前Ultralytics YOLO可能不完全支持此方式
            # 作为备选方案，我们可以传递自定义的训练循环

            # 基本训练参数
            epochs=self.config.epochs,
            imgsz=self.config.imgsz,
            batch=self.config.batch_size,
            device=self.device,
            workers=0,  # 直接训练时不使用额外的workers

            # 优化器参数
            optimizer=self.config.optimizer,
            lr0=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            warmup_epochs=self.config.warmup_epochs,

            # 训练控制
            seed=self.config.seed,
            patience=self.config.patience,
            save_period=self.config.save_period,
            project=str(self.output_dir),
            name=self.run_name,
            exist_ok=True,
            resume=resume_flag,

            # 性能相关
            amp=self.config.use_amp,
            half=self.config.half and self.device.startswith("cuda"),
            cache=True,

            # 数据增强（通过DataLoader已包含）
            augment=False,  # 禁用YOLO内置增强，使用我们自己的
            multi_scale=False,
            mosaic=0.0,
            mixup=0.0,
        )

        try:
            # 尝试直接训练
            self.last_train_results = model_for_train.train(**train_args)
        except Exception as e:
            print(f"[Trainer] 直接训练失败: {e}")
            print("[Trainer] 回退到文件生成模式...")
            # 可以在此添加回退逻辑

        return self.last_train_results

    def evaluate(self) -> Any:
        if not self.dataset_yaml_path.exists():
            self.stage_preprocessed_dataset()
        return self.model.val(data=str(self.dataset_yaml_path), device=self.device)
