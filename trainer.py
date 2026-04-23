"""
Trainer wrapper: keep custom preprocessing DataLoader, but delegate optimization/training loop to Ultralytics YOLO.train.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import os
from collections.abc import Sized
from typing import Any, Optional, cast

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Sampler, BatchSampler, SequentialSampler
from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.models.yolo.detect import DetectionTrainer

from dataset.dataset import DatasetConfig, YOLODataset, yolo_collate_fn_with_indices
from dataset.preprocess import PipelineConfig, load_config as load_preprocess_config
from ultralytics.utils.metrics import bbox_iou

# Monkey Patch for Normalized Wasserstein Distance (NWD) Loss
def bbox_nwd(box1, box2, eps=1e-6):
    """
    Calculate Normalized Wasserstein Distance between two bboxes.
    box format: (x1, y1, x2, y2)
    This patch wraps inside bbox_iou to inject NWD replacing CIoU for small targets.
    """
    # xyxy to xywh
    b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
    b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)
    
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1
    
    cx1, cy1 = b1_x1 + w1 / 2, b1_y1 + h1 / 2
    cx2, cy2 = b2_x1 + w2 / 2, b2_y1 + h2 / 2
    
    # 2D Gaussian Modeling
    c_dist = (cx1 - cx2) ** 2 + (cy1 - cy2) ** 2
    w_dist = ((w1 - w2) ** 2 + (h1 - h2) ** 2) / 4

    wass_dist = c_dist + w_dist
    
    # Normalize
    constant = 12.8  # Typically empirical scale for NWD 
    nwd = torch.exp(-torch.sqrt(wass_dist + eps) / constant)
    return nwd

original_bbox_iou = bbox_iou
def patched_bbox_iou(box1, box2, xywh=False, GIoU=False, DIoU=False, CIoU=False, eps=1e-7):
    # Calculate regular CIoU (or standard IoU depending on flags)
    iou = original_bbox_iou(box1, box2, xywh, GIoU, DIoU, CIoU, eps)
    
    # Calculate NWD to augment label assignment & CIoU loss for tiny targets
    if not xywh:
        nwd = bbox_nwd(box1, box2, eps)
        # 强制与 iou 的维度对其，避免任何意外的张量广播
        nwd = nwd.view(iou.shape)
        # NWD heavily compensates when standard IoU is mostly zero.
        return 0.5 * iou.clamp(0) + 0.5 * nwd
    return iou

# Apply the patch 
import ultralytics.utils.metrics as umetrics
umetrics.bbox_iou = patched_bbox_iou
import ultralytics.utils.loss as uloss
if hasattr(uloss, 'bbox_iou'): uloss.bbox_iou = patched_bbox_iou
import ultralytics.utils.tal as utal
if hasattr(utal, 'bbox_iou'): utal.bbox_iou = patched_bbox_iou


@dataclass
class TrainingConfig:
    # data
    train_data_dir: Path | str | list[Path | str]
    val_data_dir: Path | str | list[Path | str]
    output_dir: Path
    pre_config_path: Path

    # model
    base_model: str
    pretrained: bool
    pretrained_weights: Optional[str]
    class_names: list[str]
    num_classes: int = 1
    input_channels: int = 1
    replicate_gray_to_3ch_for_yolo_aug: bool = True

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
    models_dir: Path | str = "models"
    best_compare_metric: str = "metrics/box_map50_95"


    @staticmethod
    def to_path_list(value):
        if isinstance(value, str):
            return [Path(value)]
        return [Path(p) for p in value]

    def __post_init__(self) -> None:
        train_paths = self.to_path_list(self.train_data_dir)
        if not train_paths:
            raise ValueError("[TrainingConfig] train_data_dir is empty")
        self.train_data_dir = train_paths

        val_paths = self.to_path_list(self.val_data_dir)
        if not val_paths:
            raise ValueError("[TrainingConfig] val_data_dir is empty")
        self.val_data_dir = val_paths

        self.output_dir = Path(self.output_dir)
        self.pre_config_path = Path(self.pre_config_path)
        self.models_dir = Path(self.models_dir)
        self.class_names = self.class_names or ["target"]

    def effective_input_channels(self) -> int:
        base = max(1, int(self.input_channels))
        # Use 3-channel tensors for YOLO backbones initialized with RGB pretrained weights.
        if base == 1 and self.replicate_gray_to_3ch_for_yolo_aug:
            return 3
        return base

class DataLoaderCompatibleTrainer(DetectionTrainer):
    """扩展的YOLO训练器，支持直接传入DataLoader"""
    def __init__(self, cfg=None, overrides=None, _callbacks=None):
        custom_overrides = dict(overrides or {})
        self.custom_train_loader = YOLOLoaderAdapter(custom_overrides.pop("train_loader"))
        self.custom_val_loader = YOLOLoaderAdapter(custom_overrides.pop("val_loader"))
        if cfg is None:
            super().__init__(overrides=custom_overrides, _callbacks=_callbacks)
        else:
            super().__init__(cfg=cfg, overrides=custom_overrides, _callbacks=_callbacks)

    def get_dataset(self):
        """绕过底层yaml文件检查，直接返回虚拟数据集信息供YOLO构建Loss和Metrics"""
        active_loader = self.custom_train_loader or self.custom_val_loader
        if active_loader is not None and getattr(self, "args", None):
            train_dataset = active_loader.dataset
            dataset_names = getattr(train_dataset, "names", {0: "target"})
            dataset_nc = getattr(train_dataset, "nc", 1)
            dataset_channels = int(getattr(train_dataset, "channels", 1))
            dataset_path = getattr(train_dataset, "data_root", Path(".")).__str__()
            
            return {
                "path": dataset_path,
                "train": "dummy_train",
                "val": "dummy_val",
                "nc": dataset_nc,
                "names": dataset_names,
                "channels": dataset_channels,
            }
        return super().get_dataset()

    def get_dataloader(self, dataset_path: str, batch_size: int = 16, rank: int = 0, mode: str = "train"):
        """重写数据加载器获取"""
        if mode == "train" and self.custom_train_loader is not None:
            return self.custom_train_loader
        if mode == "val" and self.custom_val_loader is not None:
            return self.custom_val_loader
        return super().get_dataloader(dataset_path, batch_size, rank, mode)

    def plot_training_labels(self):
        """Skip label summary plotting when the custom dataset does not expose Ultralytics-style `labels`."""
        train_loader = getattr(self, "train_loader", None)
        dataset = getattr(train_loader, "dataset", None)
        if dataset is None or not hasattr(dataset, "labels"):
            print("[Trainer] skip plot_training_labels: custom dataset has no `labels` cache")
            return
        return super().plot_training_labels()


class DataLoaderCompatibleValidator(DetectionValidator):
    """Standalone validator that reuses the custom validation DataLoader."""

    def __init__(self, dataloader=None, save_dir=None, args=None, _callbacks=None):
        custom_args = dict(args or {})
        custom_loader = custom_args.pop("val_loader", None)
        dataloader = dataloader or custom_loader
        if dataloader is not None and not isinstance(dataloader, YOLOLoaderAdapter):
            dataloader = YOLOLoaderAdapter(cast(DataLoader, dataloader))
        self._logged_custom_val_loader = False
        super().__init__(dataloader=dataloader, save_dir=save_dir, args=custom_args, _callbacks=_callbacks)

    def get_dataloader(self, dataset_path, batch_size=16):
        if self.dataloader is not None:
            if not self._logged_custom_val_loader:
                print("[Validator] use custom val_loader for standalone evaluation")
                self._logged_custom_val_loader = True
            return self.dataloader
        return super().get_dataloader(dataset_path, batch_size)


class YOLOLoaderAdapter:
    """Convert the project's custom DataLoader output into Ultralytics batch dicts."""

    def __init__(self, loader: DataLoader):
        self.loader = loader
        self.dataset = loader.dataset

    def __len__(self) -> int:
        return len(self.loader)

    def __iter__(self):
        for batch in self.loader:
            if isinstance(batch, dict):
                yield batch
                continue
            yield self._convert_batch(batch)

    def _convert_batch(self, batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> dict[str, Any]:
        if len(batch) != 3:
            raise ValueError(f"expected (images, targets, sample_indices), got batch with len={len(batch)}")

        images, targets, sample_indices = batch
        if not isinstance(images, torch.Tensor) or not isinstance(targets, torch.Tensor):
            raise TypeError("custom dataloader batch must contain torch tensors for images and targets")

        if images.numel() > 0 and float(images.max().detach().cpu()) <= 1.5:
            images = images * 255.0

        sample_indices_list = (
            sample_indices.detach().cpu().tolist() if isinstance(sample_indices, torch.Tensor) else list(sample_indices)
        )
        img_h, img_w = (int(images.shape[2]), int(images.shape[3])) if images.ndim >= 4 else (0, 0)

        metas = [self._get_sample_meta(int(idx), (img_h, img_w)) for idx in sample_indices_list]
        batch_idx = targets[:, 0].to(torch.long) if targets.numel() else torch.zeros((0,), dtype=torch.long)
        cls = targets[:, 1:2].to(torch.float32) if targets.numel() else torch.zeros((0, 1), dtype=torch.float32)
        bboxes = targets[:, 2:6].to(torch.float32) if targets.numel() else torch.zeros((0, 4), dtype=torch.float32)

        return {
            "img": images,
            "batch_idx": batch_idx,
            "cls": cls,
            "bboxes": bboxes,
            "im_file": [meta["im_file"] for meta in metas],
            "ori_shape": [meta["ori_shape"] for meta in metas],
            "resized_shape": [meta["resized_shape"] for meta in metas],
            "ratio_pad": [meta["ratio_pad"] for meta in metas],
        }

    def _get_sample_meta(self, idx: int, image_shape: tuple[int, int]) -> dict[str, Any]:
        if hasattr(self.dataset, "get_yolo_sample_meta"):
            return self.dataset.get_yolo_sample_meta(idx, image_shape=image_shape)
        target_h, target_w = image_shape
        return {
            "im_file": f"sample_{idx}",
            "ori_shape": (target_h, target_w),
            "resized_shape": (target_h, target_w),
            "ratio_pad": ((1.0, 1.0), (0.0, 0.0)),
        }

    def close(self) -> None:
        if hasattr(self.loader, "dataset") and hasattr(self.loader.dataset, "close"):
            self.loader.dataset.close()

    def reset(self) -> None:
        """Compatibility hook for Ultralytics dataloader interface."""
        if hasattr(self.loader, "reset"):
            self.loader.reset()

    def __getattr__(self, item: str) -> Any:
        return getattr(self.loader, item)


class SequenceBatchSampler(Sampler[list[int]]):
    """
    Shuffles the order of sequences but yields consecutive frames grouped by sequences.
    This effectively resolves IO issues: continuous frames within the same sequence
    hit the cache efficiently, and `DataLoader` gets enough items in multithreaded workers.
    """
    def __init__(self, dataset: YOLODataset, batch_size: int, drop_last: bool, shuffle: bool = True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle

        # Precompute boundaries for each sequence
        self.sequence_indices: list[list[int]] = []
        cur_idx = 0
        for seq in dataset.sequences:
            n_frames = len(seq)
            self.sequence_indices.append(list(range(cur_idx, cur_idx + n_frames)))
            cur_idx += n_frames

    def __iter__(self):
        seq_order = list(range(len(self.sequence_indices)))
        if self.shuffle:
            np.random.shuffle(seq_order)

        batch = []
        for seq_idx in seq_order:
            for idx in self.sequence_indices[seq_idx]:
                batch.append(idx)
                if len(batch) == self.batch_size:
                    yield batch
                    batch = []
        if len(batch) > 0 and not self.drop_last:
            yield batch

    def __len__(self) -> int:
        if self.drop_last:
            return len(self.dataset) // self.batch_size
        else:
            return (len(self.dataset) + self.batch_size - 1) // self.batch_size


class CustomTrainer:
    """Use EnhancedYOLODataset/DataLoader to preprocess frames, then call YOLO.train."""

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = self._resolve_device(config.device)
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)

        self.preprocess_config: PipelineConfig = load_preprocess_config(config.pre_config_path)
        self.train_loader, self.val_loader = self.create_dataloaders()
        self.model_source = self._resolve_model_source()
        self.model = YOLO(self.model_source)

        self.output_dir = config.output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.run_name = config.run_name or datetime.now().strftime("exp_%Y%m%d_%H%M%S")
        self.staging_root = self.output_dir / "staging" / self.run_name
        self.dataset_yaml_path = self.staging_root / "dataset.yaml"
        self.last_train_results: Any = None
        self.last_val_results: Any = None
        self.last_val_metrics: dict[str, float] = {}

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


    def _make_dataset(self, image_dirs: list[Path], mode: str) -> YOLODataset:
        dataset = YOLODataset(
            DatasetConfig(
                images_dir=image_dirs,
                labels_dir=image_dirs,
                preprocess_config=self.preprocess_config,
                target_size=(self.config.imgsz, self.config.imgsz),
            ),
            mode=mode,
        )
        dataset.nc = self.config.num_classes
        dataset.names = {i: name for i, name in enumerate(self.config.class_names)}
        dataset.channels = self.config.effective_input_channels()
        dataset.data_root = image_dirs[0] if image_dirs else Path(".")
        return dataset

    def _make_loader(self, dataset: YOLODataset, shuffle: bool = False) -> DataLoader:
        batch_sampler = SequenceBatchSampler(
            dataset, batch_size=self.config.batch_size, drop_last=False, shuffle=shuffle
        )
        return DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            num_workers=self.config.workers,
            pin_memory=self.device.startswith("cuda"),
            collate_fn=yolo_collate_fn_with_indices,
        )

    def create_dataloaders(self) -> tuple[DataLoader, DataLoader]:
        train_dataset = self._make_dataset(self.config.train_data_dir, mode="train")
        val_dataset = self._make_dataset(self.config.val_data_dir, mode="val")
        return self._make_loader(train_dataset, shuffle=True), self._make_loader(val_dataset, shuffle=False)

    def stage_dataset_yaml(self) -> Path:
        """不再生成实际文件，只返回一个虚拟路径用于兼容性"""
        self.staging_root.mkdir(parents=True, exist_ok=True)
        (self.staging_root / "train").mkdir(parents=True, exist_ok=True)
        (self.staging_root / "val").mkdir(parents=True, exist_ok=True)

        dataset_yaml = {
            "path": str(self.staging_root.resolve()),
            "train": "train",
            "val": "val",
            "channels": int(self.config.effective_input_channels()),
            "nc": int(self.config.num_classes),
            "names": self.config.class_names,
        }

        self.dataset_yaml_path.parent.mkdir(parents=True, exist_ok=True)
        with self.dataset_yaml_path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(dataset_yaml, f, sort_keys=False, allow_unicode=False)
        return self.dataset_yaml_path

    def train(self, resume_from: Optional[Path] = None) -> Any:
        resume_flag: bool = False
        if resume_from and Path(resume_from).exists():
            self.model = YOLO(resume_from)
            resume_flag = True
            print(f"[Trainer] resume from: {resume_from}")
        elif resume_from:
            print(f"[Trainer] not found: {resume_from}")

        dataset_yaml = self.stage_dataset_yaml()

        train_args = dict(
            # 基本训练参数
            epochs=self.config.epochs,
            imgsz=self.config.imgsz,
            batch=self.config.batch_size,
            device=self.device,
            workers=self.config.workers,
            data=str(dataset_yaml),

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
            cache=self.config.cache,
            plots=False,
            close_mosaic=0,

            # 数据增强（交由YOLO）
            augment=self.config.augment,
            multi_scale=self.config.multi_scale,
            mosaic=float(self.config.mosaic),
            mixup=float(self.config.mixup),
            # Redundant safety: for infrared grayscale, disable color jitter.
            hsv_h=0.0 if self.config.input_channels == 1 else float(self.config.hsv_h),
            hsv_s=0.0 if self.config.input_channels == 1 else float(self.config.hsv_s),
            hsv_v=0.0 if self.config.input_channels == 1 else float(self.config.hsv_v),
            trainer=DataLoaderCompatibleTrainer,
            train_loader=self.train_loader,
            val_loader=self.val_loader,
        )
        self.last_train_results = self.model.train(**train_args)
        return self.last_train_results

    def _resolve_eval_model(self, model_path: Optional[str] = None) -> tuple[YOLO, str]:
        if model_path and Path(model_path).exists():
            return YOLO(model_path), str(model_path)
        print(f"[Trainer] evaluate model path not found, fallback to in-memory model: {model_path}")

        best_path = Path("models/best.pt")
        last_path = Path("models/last.pt")
        if best_path.exists():
            return YOLO(str(best_path)), str(best_path)
        if last_path.exists():
            return YOLO(str(last_path)), str(last_path)
        return self.model, str(self.model_source)

    @staticmethod
    def _summarize_val_results(results: Any) -> dict[str, float]:
        summary: dict[str, float] = {}
        if results is None:
            return summary

        results_dict = getattr(results, "results_dict", None)
        if isinstance(results_dict, dict):
            for key, value in results_dict.items():
                try:
                    summary[str(key)] = float(value)
                except (TypeError, ValueError):
                    continue

        box_metrics = getattr(results, "box", None)
        for attr_name, output_name in (
            ("map", "metrics/box_map50_95"),
            ("map50", "metrics/box_map50"),
            ("map75", "metrics/box_map75"),
            ("mp", "metrics/precision"),
            ("mr", "metrics/recall"),
        ):
            value = getattr(box_metrics, attr_name, None) if box_metrics is not None else None
            if value is not None:
                try:
                    summary[output_name] = float(value)
                except (TypeError, ValueError):
                    pass
        return summary

    def evaluate(self, model_path: Optional[str] = None) -> Any:
        val_dataset = getattr(self.val_loader, "dataset", None)
        val_size = len(val_dataset) if isinstance(val_dataset, Sized) else -1
        if val_size == 0:
            print("[Trainer] validation skipped: val dataset is empty")
            self.last_val_results = None
            self.last_val_metrics = {}
            return None
        eval_model = self._resolve_eval_model(model_path)
        print(f"[Trainer] evaluate model={model_path}")
        print(f"[Trainer] validation images={val_size}")
        dataset_yaml = self.stage_dataset_yaml()

        val_args = dict(
            data=str(dataset_yaml),
            imgsz=self.config.imgsz,
            batch=self.config.batch_size,
            device=self.device,
            workers=self.config.workers,
            half=self.config.half and self.device.startswith("cuda"),
            split="val",
            project=str(self.output_dir),
            name=f"{self.run_name}_val",
            exist_ok=True,
            plots=False,
            save_json=False,
            save_txt=False,
            val_loader=self.val_loader,
        )
        self.last_val_results = eval_model.val(validator=DataLoaderCompatibleValidator, **val_args)
        self.last_val_metrics = self._summarize_val_results(self.last_val_results)
        if self.last_val_metrics:
            print("[Trainer] validation summary:")
            for key in sorted(self.last_val_metrics):
                print(f"  - {key}: {self.last_val_metrics[key]:.6f}")
        else:
            print("[Trainer] validation finished, but no numeric metrics were returned")
        return self.last_val_results

