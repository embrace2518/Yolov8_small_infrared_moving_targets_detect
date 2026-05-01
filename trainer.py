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

from ultralytics.utils import RANK
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
from utils.logging import get_logger

logger = get_logger(__name__)

# ============================================================
# Loss functions: NWD for label assignment + WIoU for bbox loss
# ============================================================



def bbox_nwd(box1, box2, eps=1e-6):
    """Normalized Wasserstein Distance for small target label assignment."""
    b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
    b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1
    cx1, cy1 = b1_x1 + w1 / 2, b1_y1 + h1 / 2
    cx2, cy2 = b2_x1 + w2 / 2, b2_y1 + h2 / 2
    c_dist = (cx1 - cx2) ** 2 + (cy1 - cy2) ** 2
    w_dist = ((w1 - w2) ** 2 + (h1 - h2) ** 2) / 4
    wass_dist = c_dist + w_dist
    constant = 12.8
    nwd = torch.exp(-torch.sqrt(wass_dist + eps) / constant)
    return nwd


def bbox_wiou(box1, box2, xywh=False, eps=1e-7):
    """
    WIoU v3: dynamic non-monotonic focusing mechanism.
    box format: (x1, y1, x2, y2) unless xywh=True.
    Reference: https://arxiv.org/abs/2301.10051
    """
    if xywh:
        (cx1, cy1, w1, h1), (cx2, cy2, w2, h2) = box1.chunk(4, -1), box2.chunk(4, -1)
        b1_x1, b1_y1 = cx1 - w1 / 2, cy1 - h1 / 2
        b1_x2, b1_y2 = cx1 + w1 / 2, cy1 + h1 / 2
        b2_x1, b2_y1 = cx2 - w2 / 2, cy2 - h2 / 2
        b2_x2, b2_y2 = cx2 + w2 / 2, cy2 + h2 / 2
    else:
        b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
        b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)
        w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1
        w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1
        cx1, cy1 = (b1_x1 + b1_x2) / 2, (b1_y1 + b1_y2) / 2
        cx2, cy2 = (b2_x1 + b2_x2) / 2, (b2_y1 + b2_y2) / 2

    # IoU
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)
    union = w1 * h1 + w2 * h2 - inter + eps
    iou = inter / union

    # Enclosing box (for distance metric)
    enclose_x1 = torch.min(b1_x1, b2_x1)
    enclose_y1 = torch.min(b1_y1, b2_y1)
    enclose_x2 = torch.max(b1_x2, b2_x2)
    enclose_y2 = torch.max(b1_y2, b2_y2)
    enclose_w = enclose_x2 - enclose_x1
    enclose_h = enclose_y2 - enclose_y1

    # Center distance (R_WIoU = exp((c^2) / (wg^2 + hg^2)))
    c2 = (cx1 - cx2) ** 2 + (cy1 - cy2) ** 2
    diagonal2 = enclose_w ** 2 + enclose_h ** 2 + eps
    scale_factor = (w2 / w1.exp().clamp(min=1e-7)).detach()  # prevent extreme
    wiou = iou - (scale_factor * c2 / diagonal2)

    # Focusing mechanism: r = beta / (delta * alpha ^ (beta - delta))
    # Simplified: r = exp(beta / tau) where beta = IoU - avg(IoU)
    iou_mean = iou.detach().mean().clamp(min=1e-3)
    beta = (iou.detach() / iou_mean).clamp(max=50.0)  # prevent overflow
    r = torch.exp(beta / 3.0).clamp(max=10.0)  # tau=3.0 from paper
    wiou = wiou * (1 - 1 / (1 + r))  # down-weight easy samples

    return wiou.clamp(min=-1.0, max=3.0)


original_bbox_iou = bbox_iou


def patched_bbox_iou(box1, box2, xywh=False, GIoU=False, DIoU=False, CIoU=False, eps=1e-7):
    # Calculate standard IoU (with CIoU if requested)
    iou = original_bbox_iou(box1, box2, xywh, GIoU, DIoU, CIoU, eps)

    # Validation / eval mode: return standard IoU for fair mAP calculation
    in_validation = getattr(torch, '_v2_validation_in_progress', False)
    if in_validation or not torch.is_grad_enabled():
        return iou

    # ----- Path 1: NWD for label assignment (TAL) -----
    # TAL calls bbox_iou with CIoU=False, xywh=False → NWD blend
    if not xywh and not CIoU:
        nwd = bbox_nwd(box1, box2, eps)
        nwd = nwd.view(iou.shape)
        nwd = torch.nan_to_num(nwd, nan=0.0, posinf=0.0, neginf=0.0)
        return 0.5 * iou.clamp(0) + 0.5 * nwd

    # ----- Path 2: WIoU for bbox regression loss -----
    # bbox_loss calls bbox_iou with CIoU=True → use WIoU when enabled
    if CIoU and patched_bbox_iou._use_wiou:
        wiou = bbox_wiou(box1, box2, xywh, eps)
        return wiou

    return iou

# Default: WIoU disabled until explicitly enabled by TrainingConfig
patched_bbox_iou._use_wiou = False

# Apply monkey patches
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
    clip_grad: float = 1.0

    # ===== 预处理评估开关 =====
    # enable_preprocess_eval: 开启后，训练/验证时会额外创建一组"带预处理"的 DataLoader，
    #                        在验证时同时对原始图像和预处理图像做评估，输出对比结果。
    # 注意：实际预处理是否生效取决于 dataset_config.yaml 中的 enable_preprocess。
    #       这个开关的作用是：即使训练时关闭预处理，也能在验证时临时开启来看效果。
    enable_preprocess_eval: bool = False

    # ===== IR-YOLOv8n 改进开关 =====
    use_ir_model: bool = False      # True → 使用 models/ir_yolov8n.yaml + P2检测层
    model_yaml: str = "models/yolov8.yaml"  # 模型结构 YAML 路径
    wiou_loss: bool = True          # True → 回归损失用 WIoU（NWD 标签分配始终保留）
    copy_paste: float = 0.3         # Copy-Paste 增强概率（0=关闭）

    # ===== 两阶段训练 =====
    two_stage: bool = False         # True → 先训标准YOLOv8n基线，再微调IR-YOLOv8n
    stage1_epochs: int = 60         # 阶段1：基线的epoch数
    stage1_run_name: Optional[str] = None  # 阶段1的run_name，可选


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
        self.class_names = self.class_names or ["target"]

    def effective_input_channels(self) -> int:
        base = max(1, int(self.input_channels))
        # Use 3-channel tensors for YOLO backbones initialized with RGB pretrained weights.
        if base == 1 and self.replicate_gray_to_3ch_for_yolo_aug:
            return 3
        return base

class DataLoaderCompatibleTrainer(DetectionTrainer):
    """扩展的YOLO训练器，支持直接传入DataLoader + 自定义梯度裁剪 + 改架构时禁止ckpt恢复"""

    # Slot for pre-loaded DetectionModel from CustomTrainer (one-shot handoff)
    _pretrained_detection_model = None

    def __init__(self, cfg=None, overrides=None, _callbacks=None):
        custom_overrides = dict(overrides or {})
        self.custom_train_loader = YOLOLoaderAdapter(custom_overrides.pop("train_loader"))
        self.custom_val_loader = YOLOLoaderAdapter(custom_overrides.pop("val_loader"))
        # 提取自定义 clip_grad 参数（默认 1.0）
        self.clip_grad = float(custom_overrides.pop("clip_grad", 1.0))
        if cfg is None:
            super().__init__(overrides=custom_overrides, _callbacks=_callbacks)
        else:
            super().__init__(cfg=cfg, overrides=custom_overrides, _callbacks=_callbacks)

    def optimizer_step(self):
        """
        覆写 optimizer_step：应用自定义梯度裁剪 + NaN/Inf 权重清理。

        Ultralytics 原始实现：
          1. scaler.unscale_
          2. clip_grad_norm_(max_norm=10.0)  ← 对小目标太弱
          3. scaler.step / scaler.update / zero_grad
          4. ema.update(model)               ← 没有 NaN 防护

        修复点：
          - 使用 config 中的 clip_grad 值（默认 1.0）
          - 在 step() 之后立即清理 NaN/Inf 权重
          - 在 ema.update() 之前也清理 ema 自身
        """
        self.scaler.unscale_(self.optimizer)
        # 使用可配置的 clip_grad 值（配置文件 clip_grad: 1.0）
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.clip_grad)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad()

        # ===== NaN/Inf 防护 =====
        # 1. 清理 model 权重中的 NaN/Inf（防止污染 EMA）
        with torch.no_grad():
            for p in self.model.parameters():
                if p.is_floating_point():
                    mask = torch.isfinite(p)
                    if not mask.all():
                        p.data.masked_fill_(~mask, 0.0)

        # 2. 清理 EMA shadow 中的 NaN/Inf
        if self.ema:
            with torch.no_grad():
                for p in self.ema.ema.parameters():
                    if p.is_floating_point():
                        mask = torch.isfinite(p)
                        if not mask.all():
                            p.data.masked_fill_(~mask, 0.0)
            self.ema.update(self.model)

    def get_validator(self):
        """Override to use our DataLoaderCompatibleValidator during epoch validation"""
        self.loss_names = "box_loss", "cls_loss", "dfl_loss"
        val_args = vars(self.args).copy() if hasattr(self.args, '__dict__') else getattr(self, 'args', {}).copy()

        # force disable YOLO default plots which locks up processes
        val_args["plots"] = False
        # disable TTA: IR-YOLOv8n's multi-scale Concat paths misalign under augment flips/scales
        val_args["augment"] = False
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
            logger.info("skip plot_training_labels: custom dataset has no `labels` cache")
            return
        return super().plot_training_labels()

    def setup_model(self):
        """Override: inject pre-loaded IR-YOLOv8n model, prevent ckpt reload.

        Flow:
          1. CustomTrainer loads YAML → builds YOLO → loads pretrained → stores DetectionModel
          2. CustomTrainer sets class-level _pretrained_detection_model
          3. Model.train() creates DataLoaderCompatibleTrainer → calls setup_model()
          4. setup_model() detects the pre-loaded model and injects it
          5. Returns None so resume_training() doesn't load stale optimizer state

        For resume (`.pt` path): fall through to default which handles
        weight loading and ckpt return for resume_training().
        """
        if isinstance(self.model, torch.nn.Module):
            return  # already loaded (e.g. from resume)

        # Consume pre-loaded model from CustomTrainer (one-shot)
        pt_model = getattr(self.__class__, "_pretrained_detection_model", None)
        if pt_model is not None:
            self.model = pt_model
            self.__class__._pretrained_detection_model = None  # consume once
            return  # ckpt=None → skip resume_training

        # Resume from a .pt checkpoint → use default behavior (load weights + ckpt)
        if isinstance(self.model, (str, Path)) and str(self.model).lower().endswith(".pt"):
            return super().setup_model()

        # YAML path with no pre-loaded weights → build fresh
        self.model = self.get_model(cfg=str(self.model), weights=None, verbose=RANK in {-1, 0})
        return None


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
                logger.info("Validator: use custom val_loader for standalone evaluation")
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

        # IR-YOLOv8n: load from YAML then load pretrained weights
        if self.config.use_ir_model:
            from models.custom_modules import ECA, GSConv, C2f_GS  # noqa: ensure registration
            self.model = YOLO(self.config.model_yaml)
            self._ir_pretrained_loaded = False
            if self.config.pretrained and self.config.pretrained_weights:
                self._load_pretrained_into_ir(self.config.pretrained_weights)
                self._ir_pretrained_loaded = True
                # Hand off the built DetectionModel to the trainer via class-level slot
                DataLoaderCompatibleTrainer._pretrained_detection_model = self.model.model
        else:
            self.model = YOLO(self.model_source)

        # Set WIoU flag on the patched function (scope it to the function, not the module)
        patched_bbox_iou._use_wiou = self.config.wiou_loss

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

        logger.info("device=%s", self.device)
        logger.info("model=%s", self.model_source)
        self._log_runtime_env()
        train_size = len(self.train_loader.dataset) if isinstance(self.train_loader.dataset, Sized) else -1
        val_size = len(self.val_loader.dataset) if isinstance(self.val_loader.dataset, Sized) else -1
        logger.info("train images=%s", train_size)
        logger.info("val images=%s", val_size)


    def _log_runtime_env(self) -> None:
        logger.info("torch=%s", torch.__version__)
        logger.info("cuda_available=%s, cuda_count=%s", torch.cuda.is_available(), torch.cuda.device_count())
        if torch.cuda.is_available():
            try:
                idx = torch.cuda.current_device()
                logger.info("gpu=%s", torch.cuda.get_device_name(idx))
            except Exception as exc:
                logger.info("failed to query GPU name: %s", exc)
        logger.info("CUDA_LAUNCH_BLOCKING=%s", os.environ.get('CUDA_LAUNCH_BLOCKING'))

    @staticmethod
    def _resolve_device(requested: str) -> str:
        if requested.startswith("cuda") and not torch.cuda.is_available():
            logger.warning("CUDA unavailable, fallback to CPU")
            return "cpu"
        return requested

    def _resolve_model_source(self) -> str:
        if self.config.pretrained and self.config.pretrained_weights:
            return self.config.pretrained_weights
        return self.config.model_yaml if self.config.use_ir_model else self.config.base_model


    def _load_pretrained_into_ir(self, weights_path: str | Path) -> None:
        """Load pretrained weights into IR-YOLOv8n, logging matched / mismatched layers.

        Uses intersect_dicts + load_state_dict(strict=False) directly instead of
        YOLO.load(), because YOLO.load() also caches the full checkpoint (including
        optimizer) into self.ckpt, which later triggers a mismatched optimizer
        resume error when the architecture changes (baseline → IR).
        """
        import torch
        from ultralytics.nn.tasks import intersect_dicts

        ckpt = torch.load(str(weights_path), map_location="cpu", weights_only=False)
        src_model = ckpt.get("model") or ckpt.get("ema")
        if src_model is None:
            raise ValueError(f"[Trainer] checkpoint {weights_path} has no model or ema state")
        src_sd = src_model.float().state_dict()
        dst_sd = self.model.model.state_dict()

        matched = []
        shape_mismatch = []
        for k in dst_sd:
            if k in src_sd:
                if dst_sd[k].shape == src_sd[k].shape:
                    matched.append(k)
                else:
                    shape_mismatch.append((k, src_sd[k].shape, dst_sd[k].shape))

        # Load only intersecting weights, skip optimizer/epoch state
        updated_csd = intersect_dicts(src_sd, dst_sd)
        self.model.model.load_state_dict(updated_csd, strict=False)

        # Report
        n_total = len(dst_sd)
        n_matched = len(matched)
        n_ir_only = n_total - len(set(matched) | {k for k, _, _ in shape_mismatch})
        logger.info("IR-YOLOv8n ← %s", Path(weights_path).name)
        logger.info("%s/%s keys transferred (%s%%)", n_matched, n_total, n_matched * 100 // n_total)
        if shape_mismatch:
            logger.info("%s keys shape mismatch (skipped):", len(shape_mismatch))
            for k, s, d in shape_mismatch[:5]:
                logger.info("  %s: src=%s dst=%s", k, tuple(s), tuple(d))
        if n_ir_only > 0:
            logger.info("%s IR-only keys (random init)", n_ir_only)

    def _make_dataset(self, image_dirs: list[Path], mode: str) -> YOLODataset:
        dataset = YOLODataset(
            DatasetConfig(
                images_dir=image_dirs,
                labels_dir=image_dirs,
                unified_config=self.unified_config,
                target_size=(self.config.imgsz, self.config.imgsz),
            ),
            mode=mode,
            copy_paste_prob=float(self.config.copy_paste),
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
    def _check_resume_checkpoint(checkpoint_path: Path) -> tuple[bool, dict]:
        """Return (can_full_resume, checkpoint_dict).

        can_full_resume=True means the checkpoint has epoch & optimizer state
        for true YOLO resume.  Otherwise only model weights are usable.
        """
        if not checkpoint_path.exists() or not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"[Trainer] resume checkpoint not found: {checkpoint_path}. "
                "Please pass an existing last.pt path."
            )

        if checkpoint_path.suffix.lower() != ".pt":
            raise ValueError(f"[Trainer] resume file must be a .pt checkpoint, got: {checkpoint_path}")

        try:
            checkpoint = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
        except TypeError:
            checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
        except Exception as exc:
            raise ValueError(f"[Trainer] failed to read checkpoint: {checkpoint_path} ({exc})") from exc

        if not isinstance(checkpoint, dict):
            raise ValueError(f"[Trainer] invalid checkpoint format: {checkpoint_path}")

        epoch = checkpoint.get("epoch")
        optimizer = checkpoint.get("optimizer")
        can_full_resume = (epoch is not None and epoch >= 0 and optimizer is not None)
        return can_full_resume, checkpoint

    def _build_train_args(self, epochs: int, lr0: float, patience: int, name: str) -> dict:
        """Build the common training args dict shared across all training entry points.

        Resume path has too-few params and uses a separate dict inline.
        """
        return dict(
            epochs=epochs,
            imgsz=self.config.imgsz,
            batch=self.config.batch_size,
            device=self.device,
            workers=self.config.workers,
            data=str(self.stage_dataset_yaml()),

            # 优化器参数
            optimizer=self.config.optimizer,
            lr0=lr0,
            weight_decay=self.config.weight_decay,
            warmup_epochs=self.config.warmup_epochs,

            # 训练控制
            seed=self.config.seed,
            patience=patience,
            save_period=self.config.save_period,
            project=str(self.output_dir.absolute()),
            name=name,
            exist_ok=True,

            # 性能相关
            amp=self.config.use_amp,
            half=self.config.half and self.device.startswith("cuda"),
            cache=self.config.cache,
            plots=False,
            close_mosaic=0,

            # 数据增强
            augment=self.config.augment,
            multi_scale=self.config.multi_scale,
            mosaic=float(self.config.mosaic),
            mixup=float(self.config.mixup),
            hsv_h=0.0 if self.config.input_channels == 1 else float(self.config.hsv_h),
            hsv_s=0.0 if self.config.input_channels == 1 else float(self.config.hsv_s),
            hsv_v=0.0 if self.config.input_channels == 1 else float(self.config.hsv_v),

            # 梯度裁剪
            clip_grad=float(self.config.clip_grad),

            trainer=DataLoaderCompatibleTrainer,
            train_loader=self.train_loader,
            val_loader=self.val_loader,
        )

    def _sanitize_ema_before_save(self, trainer) -> None:
        """Callback: zero-out NaN/Inf in EMA parameters before checkpoint save."""
        ema = getattr(trainer, "ema", None)
        if ema is None:
            return
        ema_model = ema.ema  # ModelEMA stores the model in .ema
        if ema_model is None:
            return
        with torch.no_grad():
            for p in ema_model.parameters():
                if p.is_floating_point():
                    mask = torch.isfinite(p)
                    if not mask.all():
                        p.data.masked_fill_(~mask, 0.0)

    def train(self, resume_from: Optional[Path] = None) -> Any:
        # ===== 两阶段训练 =====
        if self.config.two_stage and self.config.use_ir_model and resume_from is None:
            return self._train_two_stage()

        resume_flag: bool = False
        if resume_from:
            resume_path = Path(resume_from)
            can_resume, _ = self._check_resume_checkpoint(resume_path)
            if can_resume:
                self.model = YOLO(resume_path)
                resume_flag = True
                logger.info("resume from: %s", resume_path)
            else:
                # checkpoint lacks epoch/optimizer — use weights only, start fresh
                logger.info("checkpoint has no training state, loading weights only")
                if self.config.use_ir_model:
                    self._load_pretrained_into_ir(resume_path)
                else:
                    self.model = YOLO(resume_path)
                resume_flag = False

        dataset_yaml = self.stage_dataset_yaml()

        # Register a callback to sanitise EMA before checkpoint save
        self.model.add_callback("on_model_save", self._sanitize_ema_before_save)

        if resume_flag:
            train_args = dict(
                resume=True,
                data=str(dataset_yaml),
                epochs=self.config.epochs,  # 只允许扩展 Epoch 总数
                clip_grad=float(self.config.clip_grad),
                trainer=DataLoaderCompatibleTrainer,
                train_loader=self.train_loader,
                val_loader=self.val_loader,
            )
        else:
            train_args = self._build_train_args(
                epochs=self.config.epochs,
                lr0=self.config.learning_rate,
                patience=self.config.patience,
                name=self.run_name,
            )

        self.last_train_results = self.model.train(**train_args)

        # ===== 预处理对比评估 =====
        if self.config.enable_preprocess_eval:
            logger.info("")
            logger.info("=" * 70)
            logger.info("正在进行预处理对比评估...")
            logger.info("=" * 70)
            self._run_preprocess_comparison()

        return self.last_train_results

    def _train_two_stage(self) -> Any:
        """两阶段训练：Stage 1 = 标准YOLOv8n → Stage 2 = IR-YOLOv8n"""

        # ---------- Stage 1: 标准 YOLOv8n ----------
        logger.info("")
        logger.info("=" * 70)
        logger.info("Two-Stage: Stage 1/2: 训练标准YOLOv8n (%s epochs)", self.config.stage1_epochs)
        logger.info("=" * 70)

        # Temporarily switch to standard model
        self.model = YOLO(self.config.base_model)

        stage1_name = self.config.stage1_run_name or f"{self.run_name}_stage1"
        stage1_args = self._build_train_args(
            epochs=self.config.stage1_epochs,
            lr0=self.config.learning_rate,
            patience=int(self.config.patience / 2),
            name=stage1_name,
        )

        stage1_results = self.model.train(**stage1_args)

        # Find best stage 1 weights
        stage1_weights = self.output_dir / stage1_name / "weights" / "best.pt"
        if not stage1_weights.exists():
            stage1_weights = self.output_dir / stage1_name / "weights" / "last.pt"
        logger.info("")
        logger.info("Two-Stage: Stage 1 complete. Best weights: %s", stage1_weights)

        # ---------- Stage 2: IR-YOLOv8n ----------
        logger.info("")
        logger.info("=" * 70)
        logger.info("Two-Stage: Stage 2/2: 微调IR-YOLOv8n (%s epochs)", self.config.epochs)
        logger.info("=" * 70)

        # Re-init for IR-YOLOv8n
        from models.custom_modules import ECA, GSConv, C2f_GS  # noqa: ensure registration
        self.model = YOLO(self.config.model_yaml)
        self._load_pretrained_into_ir(str(stage1_weights))
        # Hand off the built DetectionModel to the trainer
        DataLoaderCompatibleTrainer._pretrained_detection_model = self.model.model

        self.model.add_callback("on_model_save", self._sanitize_ema_before_save)

        # Update WIoU on the patched function
        patched_bbox_iou._use_wiou = self.config.wiou_loss

        stage2_args = self._build_train_args(
            epochs=self.config.epochs,
            lr0=self.config.learning_rate * 0.1,  # 微调使用更低学习率
            patience=self.config.patience,
            name=self.run_name,
        )

        self.last_train_results = self.model.train(**stage2_args)

        # ===== 预处理对比评估 =====
        if self.config.enable_preprocess_eval:
            logger.info("")
            logger.info("=" * 70)
            logger.info("正在进行预处理对比评估...")
            logger.info("=" * 70)
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
        logger.info("")
        logger.info("=" * 70)
        logger.info("Preprocess Eval: 无预处理 (原始图像)...")
        logger.info("=" * 70)
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
        logger.info("")
        logger.info("=" * 70)
        logger.info("Preprocess Eval: 有预处理 (NUC+去噪+CLAHE+Gamma)...")
        logger.info("=" * 70)
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
        logger.info("对比结果已保存: %s", comparison_path)

        # 打印汇总表
        logger.info("=" * 95)
        logger.info("预处理对比评估摘要")
        logger.info("=" * 95)
        header = f"  {'指标':<25} {'无预处理':<18} {'有预处理':<18} {'差值':<12} {'提升%':<10}"
        logger.info(header)
        logger.info("  " + "-" * 85)
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
            logger.info("  %-25s %-18s %-18s %-12s %-10s%s", key, raw_str, pp_str, diff_str, pct_str, suffix)
        logger.info("=" * 95)
