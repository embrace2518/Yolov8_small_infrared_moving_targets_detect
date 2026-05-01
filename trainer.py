"""
Trainer wrapper: keep custom preprocessing DataLoader, but delegate optimization/training loop to Ultralytics YOLO.train.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import os
import json
from collections.abc import Sized
from contextlib import suppress
from typing import Any, Optional, cast

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Sampler
from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.models.yolo.detect import DetectionTrainer

from dataset.dataset import DatasetConfig, YOLODataset, yolo_collate_fn_with_indices
from dataset.preprocess import UnifiedConfig, load_config as load_preprocess_config
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
    
    # 全局开关判定：如果正在调用 NWD，需要检查是否正处于验证阶段。
    # 避免 NWD 破坏原有的评估 mAP 的 IOU 阈值标准！
    in_validation = getattr(torch, '_v2_validation_in_progress', False)
    if in_validation or not torch.is_grad_enabled():
        return iou

    # Calculate NWD ONLY to augment label assignment (TAL Assigner).
    # We strictly avoid modifying the Bbox Loss (where CIoU is typically True)
    # to prevent extreme gradient vanishing for distant initial predictions.
    if not xywh and not CIoU:
        nwd = bbox_nwd(box1, box2, eps)
        nwd = nwd.view(iou.shape)
        # NUMERICAL SAFETY: replace any NaN/Inf NWD with 0 so they don't corrupt EMA
        nwd = torch.nan_to_num(nwd, nan=0.0, posinf=0.0, neginf=0.0)
        # NWD heavily compensates when standard IoU is mostly zero in tiny targets.
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
    dataset_config_path: Path

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
    workers: int = 4
    cache: bool = True
    use_amp: bool = True
    half: bool = False
    multi_scale: bool = False  # avoid random zero-size in unstable envs

    # aug
    augment: bool = True
    mosaic: float = 0.1
    mixup: float = 0.0
    hsv_h: float = 0.015
    hsv_s: float = 0.7
    hsv_v: float = 0.4

    # control
    seed: int = 42
    patience: int = 20
    save_period: int = 20
    run_name: Optional[str] = None
    refresh_staging: bool = True
    models_dir: Path | str = "models"
    best_compare_metric: str = "metrics/box_map50_95"
    clip_grad: float = 1.0

    # ===== 预处理评估开关 =====
    # enable_preprocess_eval: 开启后，训练/验证时会额外创建一组"带预处理"的 DataLoader，
    #                        在验证时同时对原始图像和预处理图像做评估，输出对比结果。
    # 注意：实际预处理是否生效取决于 dataset_config.yaml 中的 enable_preprocess。
    #       这个开关的作用是：即使训练时关闭预处理，也能在验证时临时开启来看效果。
    enable_preprocess_eval: bool = False


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
        self.dataset_config_path = Path(self.dataset_config_path)
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

    def get_validator(self):
        """Override to use our DataLoaderCompatibleValidator during epoch validation"""
        self.loss_names = "box_loss", "cls_loss", "dfl_loss"
        val_args = vars(self.args).copy() if hasattr(self.args, '__dict__') else getattr(self, 'args', {}).copy()

        # force disable YOLO default plots which locks up processes
        val_args["plots"] = False
        # pass in custom_val_loader so the validator overrides its own dataloader
        val_args["val_loader"] = getattr(self, 'custom_val_loader', None)

        validator = DataLoaderCompatibleValidator(
            dataloader=self.test_loader if hasattr(self, 'test_loader') else self.custom_val_loader,
            save_dir=self.save_dir,
            args=val_args
        )
        return validator

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

    def __call__(self, trainer=None, model=None):
        # 开启验证阶段 NWD 的锁定标志
        setattr(torch, '_v2_validation_in_progress', True)
        try:
            return super().__call__(trainer, model)
        finally:
            setattr(torch, '_v2_validation_in_progress', False)


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
        super().__init__(None) # PyTorch Sampler initialization
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

        self.unified_config: UnifiedConfig = load_preprocess_config(config.dataset_config_path)
        self.train_loader, self.val_loader = self.create_dataloaders()
        self.model_source = self._resolve_model_source()
        self.model = YOLO(self.model_source)

        # ===== 预处理对比评估 =====
        # 已迁移至 evaluate.py，训练完成后由 _run_preprocess_comparison 统一调用。
        self.val_loader_with_preprocess = None

        self.output_dir = config.output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.run_name = config.run_name or datetime.now().strftime("exp_%Y%m%d_%H%M%S")
        self.staging_root = self.output_dir / "staging" / "current_run"
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


    def _log_runtime_env(self) -> None:
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
                unified_config=self.unified_config,
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
        train_dataset = self._make_dataset(cast(list[Path], self.config.train_data_dir), mode="train")
        val_dataset = self._make_dataset(cast(list[Path], self.config.val_data_dir), mode="val")
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


    @staticmethod
    def _assert_resume_checkpoint(checkpoint_path: Path) -> None:
        if not checkpoint_path.exists() or not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"[Trainer] resume checkpoint not found: {checkpoint_path}. "
                "Please pass an existing last.pt path."
            )

        if checkpoint_path.suffix.lower() != ".pt":
            raise ValueError(f"[Trainer] resume file must be a .pt checkpoint, got: {checkpoint_path}")

        try:
            # PyTorch>=2.6 defaults to weights_only=True which drops optimizer/epoch state.
            checkpoint = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
        except TypeError:
            # Compatibility fallback for older torch versions without weights_only argument.
            checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
        except Exception as exc:
            raise ValueError(f"[Trainer] failed to read checkpoint: {checkpoint_path} ({exc})") from exc

        if not isinstance(checkpoint, dict):
            raise ValueError(f"[Trainer] invalid checkpoint format: {checkpoint_path}")

        epoch_ok = checkpoint.get("epoch") is not None
        optimizer_ok = checkpoint.get("optimizer") is not None
        if not (epoch_ok and optimizer_ok):
            raise ValueError(
                f"[Trainer] checkpoint lacks resume state (epoch/optimizer): {checkpoint_path}. "
                "Use a true last.pt instead of best.pt for resume training."
            )

    def _sanitize_ema_before_save(self, trainer) -> None:
        """Callback: zero-out NaN/Inf in EMA parameters before checkpoint save."""
        ema = getattr(trainer, "ema", None)
        if ema is None:
            return
        with torch.no_grad():
            for p in ema.parameters():
                if p.is_floating_point():
                    mask = torch.isfinite(p)
                    if not mask.all():
                        p.data.masked_fill_(~mask, 0.0)

    def _clip_gradients(self, trainer) -> None:
        """Callback: gradient clipping on each optimizer step."""
        clip_val = self.config.clip_grad
        if clip_val > 0 and hasattr(trainer, "optimizer"):
            torch.nn.utils.clip_grad_norm_(
                (p for p in trainer.model.parameters() if p.grad is not None),
                clip_val,
            )

    def train(self, resume_from: Optional[Path] = None) -> Any:
        resume_flag: bool = False
        if resume_from:
            resume_path = Path(resume_from)
            self._assert_resume_checkpoint(resume_path)
            self.model = YOLO(resume_path)
            resume_flag = True
            print(f"[Trainer] resume from: {resume_path}")

        dataset_yaml = self.stage_dataset_yaml()

        # Register a callback to sanitise EMA before checkpoint save
        self.model.add_callback("on_model_save", self._sanitize_ema_before_save)

        if self.config.clip_grad > 0:
            self.model.add_callback("on_before_optimizer_step", self._clip_gradients)
            print(f"[Trainer] 梯度裁剪已启用: max_norm={self.config.clip_grad}")

        if resume_flag:
            train_args = dict(
                resume=True,
                data=str(dataset_yaml),
                epochs=self.config.epochs,  # 只允许扩展 Epoch 总数
                trainer=DataLoaderCompatibleTrainer,
                train_loader=self.train_loader,
                val_loader=self.val_loader,
            )
        else:
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
                project=str(self.output_dir.absolute()),
                name=self.run_name,
                exist_ok=True,

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

        # ===== 预处理对比评估 =====
        if self.config.enable_preprocess_eval:
            print("\n" + "=" * 70)
            print("[Trainer] 正在进行预处理对比评估...")
            print("=" * 70)
            self._run_preprocess_comparison()

        return self.last_train_results
    def _run_preprocess_comparison(self) -> None:
        """
        运行预处理对比评估（合并 YOLO mAP + spotGEO 竞赛指标）。
        使用 evaluate 模块统一评估逻辑，避免代码重复。
        对比结果保存至 comparison.json。
        """
        from evaluate import evaluate as run_eval

        comparison_dir = self.output_dir / self.run_name / "preprocess_comparison"
        comparison_dir.mkdir(parents=True, exist_ok=True)

        best_pt_path = str(self.output_dir / self.run_name / "weights" / "best.pt")
        val_sources = self.config.val_data_dir

        # ===== 1. 无预处理评估 (raw) =====
        print("\n" + "=" * 70)
        print("[Preprocess Eval] 无预处理 (原始图像)...")
        print("=" * 70)
        raw_metrics = run_eval(
            weights=best_pt_path,
            sources=val_sources,
            enable_preprocess=False,
            batch_size=self.config.batch_size,
            img_size=self.config.imgsz,
            output_dir=str(comparison_dir),
            run_name="raw",
            max_visualize=5,  # 少量可视化，避免过多输出
            no_save=False,
        )

        # ===== 2. 有预处理评估 (preprocessed) =====
        print("\n" + "=" * 70)
        print("[Preprocess Eval] 有预处理 (NUC+去噪+CLAHE+Gamma)...")
        print("=" * 70)
        pp_metrics = run_eval(
            weights=best_pt_path,
            sources=val_sources,
            preprocess_config_path=self.config.dataset_config_path,
            enable_preprocess=True,
            batch_size=self.config.batch_size,
            img_size=self.config.imgsz,
            output_dir=str(comparison_dir),
            run_name="preprocessed",
            max_visualize=5,
            no_save=False,
        )

        # ===== 3. 整理对比结果 =====
        def _extract_key(m: dict, k: str):
            """安全提取指标值"""
            v = m.get(k, None)
            if v is not None:
                try:
                    return float(v)
                except (ValueError, TypeError):
                    return None
            return None

        # 所有关注的指标
        metric_keys = [
            'fps', 'avg_latency_ms',
            'spotgeo_score', 'spotgeo_mse',
            'spotgeo_precision', 'spotgeo_recall', 'spotgeo_f1',
        ]

        comparison = {}
        for key in metric_keys:
            raw_v = _extract_key(raw_metrics, key)
            pp_v = _extract_key(pp_metrics, key)
            if raw_v is not None or pp_v is not None:
                entry = {}
                if raw_v is not None:
                    entry["raw"] = raw_v
                if pp_v is not None:
                    entry["preprocessed"] = pp_v
                if raw_v is not None and pp_v is not None:
                    entry["diff"] = pp_v - raw_v
                    entry["improvement_pct"] = float(f"{((pp_v - raw_v) / abs(raw_v) * 100):.2f}") if raw_v != 0 else 0.0
                comparison[key] = entry

        # 保存对比结果
        comparison_path = comparison_dir / "comparison.json"
        with comparison_path.open("w", encoding="utf-8") as f:
            json.dump(comparison, f, indent=2, ensure_ascii=False)
        print(f"\n[Preprocess Eval] 对比结果已保存: {comparison_path}")

        # 打印汇总表
        print("\n" + "=" * 95)
        print("预处理对比评估摘要")
        print("=" * 95)
        header = f"  {'指标':<25} {'无预处理':<18} {'有预处理':<18} {'差值':<12} {'提升%':<10}"
        print(header)
        print("  " + "-" * 85)
        for key in metric_keys:
            entry = comparison.get(key)
            if entry is None:
                continue
            raw_v = entry.get("raw")
            pp_v = entry.get("preprocessed")
            diff = entry.get("diff")
            pct = entry.get("improvement_pct")

            raw_str = f"{raw_v:.4f}" if raw_v is not None else "N/A"
            pp_str = f"{pp_v:.4f}" if pp_v is not None else "N/A"
            diff_str = f"{diff:+.4f}" if diff is not None else ""
            pct_str = f"{pct:+.2f}%" if pct is not None else ""
            suffix = " fps" if "fps" in key else ""
            print(f"  {key:<25} {raw_str:<18} {pp_str:<18} {diff_str:<12} {pct_str:<10}{suffix}")
        print("=" * 95)
