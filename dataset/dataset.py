from __future__ import annotations

import logging
import re
from collections import OrderedDict
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from .preprocess import ImagePreprocessor, ImageReadError, UnifiedConfig, SceneBasedNUC, read_gray_image
from .sampling import get_recursion_guard
from utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class DatasetConfig:
    images_dir: Path | str | list[Path | str]
    labels_dir: Path | str | list[Path | str]
    unified_config: UnifiedConfig
    target_size: tuple[int, int]

    cache_images: bool = True
    cache_labels: bool = True
    prefetch: bool = True
    max_labels: int = 300
    prefetch_workers: int = 2
    prefetch_size: int = 128

    true_labels_json_path: Path | str | None = None
    json_label_class_id: int = 0
    json_fallback_box_width: int = 16
    json_fallback_box_height: int = 16


@dataclass
class SequenceData:
    """单个序列的数据"""
    name: str
    frame_paths: list[Path]
    label_paths: list[Path]
    image_cache: OrderedDict[int, np.ndarray] = field(default_factory=OrderedDict)
    label_cache: OrderedDict[int, np.ndarray] = field(default_factory=OrderedDict)
    nuc: SceneBasedNUC | None = None
    last_processed_frame: int = -1

    def __len__(self) -> int:
        return len(self.frame_paths)


class YOLODataset(Dataset):

    def __init__(self, config: DatasetConfig, mode: str, copy_paste_prob: float = 0.3) -> None:
        self.config = config
        self.mode = mode
        if self.mode == "train" and self.config.unified_config.enable_augmentation:
            aug_cfg = self.config.unified_config.augmentation_config
            aug_mode = aug_cfg.get('mode', 'medium')  # 从配置读取增强等级
            from .augmentations import get_infrared_augmentation_pipeline
            self.augmentation = get_infrared_augmentation_pipeline(aug_mode, dataset=self,
                                                                    copy_paste_prob=copy_paste_prob)
        else:
            self.augmentation = None
        self.image_dirs = self._normalize_dirs(config.images_dir)
        self.label_dirs = self._normalize_dirs(config.labels_dir)
        self.target_size = config.target_size
        self.logger = logging.getLogger(__name__)

        # 创建图像预处理器
        if self.config.unified_config.enable_preprocess:
            self.preprocessor = ImagePreprocessor(config.unified_config.preprocess_config)
        else:
            self.preprocessor = None

        self.sequences = self._load_sequences()
        if not self.sequences:
            raise FileNotFoundError(f"no image files found at {config.images_dir}")

        self.samples = self._build_sample_index()

        self._read_fail_count = 0
        self._read_fail_log_limit = 20

    @staticmethod
    def _normalize_dirs(path_like: Path | str | list[Path | str]) -> list[Path]:
        if isinstance(path_like, (str, Path)):
            return [Path(path_like)]
        return [Path(p) for p in path_like]

    def _image_sort_key(self, path: Path) -> tuple[str, int, str]:
        # Precompute sort key: (parent_dir, last_number, stem)
        stem = path.stem
        numbers = list(map(int, re.findall(r'\d+', stem)))
        primary_num = numbers[-1] if numbers else -1
        return str(path.parent), primary_num, stem

    def _load_image_files(self) -> list[Path]:
        image_exts = self.config.unified_config.preprocess_config.image_exts
        image_exts_lower = {ext.lower() for ext in image_exts}
        # Single pass: collect all files, filter by extension
        all_files: list[Path] = []
        for img_dir in self.image_dirs:
            for entry in img_dir.rglob("*"):
                if entry.is_file() and entry.suffix.lower() in image_exts_lower:
                    all_files.append(entry)
        return sorted(all_files, key=self._image_sort_key)

    def _load_sequences(self) -> list[SequenceData]:
        grouped: dict[str, list[Path]] = {}
        for img_path in self._load_image_files():
            grouped.setdefault(str(img_path.parent), []).append(img_path)

        sequences: list[SequenceData] = []
        for parent_str, frame_paths in grouped.items():
            ordered_frames = sorted(frame_paths, key=self._image_sort_key)
            label_paths = [p.with_suffix(".txt") for p in ordered_frames]
            sequences.append(
                SequenceData(name=Path(parent_str).name, frame_paths=ordered_frames, label_paths=label_paths, ))
        sequences.sort(key=lambda s: s.name)
        return sequences

    def _relative_to_image_root(self, image_path: Path) -> Path:
        for root in self.image_dirs:
            with suppress(ValueError):
                return image_path.relative_to(root)
        return Path(image_path.name)

    def _build_sample_index(self) -> list[tuple[int, int]]:
        """构建数据集索引列表，每个元素为 (序列索引, 帧索引)。"""
        return [
            (seq_idx, frame_idx)
            for seq_idx, seq in enumerate(self.sequences)
            for frame_idx in range(len(seq))  # 使用 seq.__len__() 接口
        ]

    def _apply_preprocessing(self, gray_image: np.ndarray, seq: SequenceData, frame_idx: int) -> np.ndarray:
        if not self.config.unified_config.enable_preprocess:
            return gray_image
        processed = self.preprocessor.process_image(
            gray_image,
            seq_nuc=seq.nuc,
            frame_idx=frame_idx,
            last_processed_frame=seq.last_processed_frame
        )
        # 更新NUC状态追踪
        seq.last_processed_frame = frame_idx
        return processed

    def _cache_image(self, seq_idx: int, frame_idx: int, image: np.ndarray) -> None:
        if not self.config.cache_images:
            return
        seq = self.sequences[seq_idx]
        seq.image_cache[frame_idx] = image.copy()

    def _load_image(self, seq_idx: int, frame_idx: int) -> np.ndarray:
        seq = self.sequences[seq_idx]
        if self.config.cache_images:
            cached = seq.image_cache.get(frame_idx)
            if cached is not None:
                return cached.copy()

        img_path = seq.frame_paths[frame_idx]
        try:
            gray = read_gray_image(img_path)
            processed = self._apply_preprocessing(gray, seq=seq, frame_idx=frame_idx)
        except ImageReadError as e:
            self._read_fail_count += 1
            if self._read_fail_count <= self._read_fail_log_limit:
                logger.warning("read failed, use placeholder: %s", e)
                if self._read_fail_count == self._read_fail_log_limit:
                    logger.warning("too many read errors, suppressing further logs...")

            # Fallback keeps training running when single files are corrupted/unreadable.
            target_h, target_w = self.target_size
            processed = np.zeros((target_h, target_w), dtype=np.uint8)

        self._cache_image(seq_idx, frame_idx, processed)
        return processed.copy()

    def _cache_labels(self, seq_idx: int, frame_idx: int, labels: np.ndarray) -> None:
        if not self.config.cache_labels:
            return
        seq = self.sequences[seq_idx]
        seq.label_cache[frame_idx] = labels.copy()

    def _load_yolo_labels(self, seq_idx: int, frame_idx: int) -> np.ndarray:
        seq = self.sequences[seq_idx]
        if self.config.cache_labels:
            cached = seq.label_cache.get(frame_idx)
            if cached is not None:
                return cached.copy()

        label_path = seq.label_paths[frame_idx]

        if not label_path.exists():
            empty = np.zeros((0, 5), dtype=np.float32)
            self._cache_labels(seq_idx, frame_idx, empty)
            return empty

        labels = []
        with label_path.open("r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    try:
                        class_id, x_center, y_center, width, height = (float(v) for v in parts[:5])
                        if 0 <= x_center <= 1 and 0 <= y_center <= 1 and 0 <= width <= 1 and 0 <= height <= 1:
                            labels.append([class_id, x_center, y_center, width, height])
                    except ValueError:
                        continue

        if len(labels) > self.config.max_labels:
            labels = labels[: self.config.max_labels]

        parsed = np.array(labels, dtype=np.float32) if labels else np.zeros((0, 5), dtype=np.float32)
        self._cache_labels(seq_idx, frame_idx, parsed)
        return parsed

    def _resize_image(self, image: np.ndarray) -> np.ndarray:
        if self.preprocessor:
            return self.preprocessor.resize_image(image, self.target_size)
        else:
            import cv2
            target_h, target_w = self.target_size
            return cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        seq_idx, frame_idx = self.samples[idx]
        image = self._load_image(seq_idx, frame_idx)
        labels = self._load_yolo_labels(seq_idx, frame_idx)

        channels = int(getattr(self, "channels", 1))

        if channels == 3:
            # 时序堆叠: [|frame_t - frame_{t-1}|, frame_t, |frame_{t+1} - frame_t|]
            # 一前一后两个差分 + 当前帧，形成以 t 为中心的对称时序窗口
            seq_len = len(self.sequences[seq_idx])
            prev_idx = max(0, frame_idx - 1)
            next_idx = min(seq_len - 1, frame_idx + 1)

            image_prev = self._load_image(seq_idx, prev_idx) if prev_idx != frame_idx else image
            image_next = self._load_image(seq_idx, next_idx) if next_idx != frame_idx else image

            image = self._resize_image(image)
            image_prev = self._resize_image(image_prev)
            image_next = self._resize_image(image_next)

            diff_prev = np.abs(image.astype(np.int16) - image_prev.astype(np.int16)).astype(np.uint8)
            diff_next = np.abs(image_next.astype(np.int16) - image.astype(np.int16)).astype(np.uint8)

            stacked = np.stack([diff_prev, image, diff_next], axis=0)
        else:
            image = self._resize_image(image)
            stacked = image[np.newaxis, ...]  # Shape: 1 x H x W

        # 优化：在 DataLoader 端加入强力的数据增强（解决 Bypass YOLO Transformer 的问题），保持时序一致性
        if self.mode == "train" and self.augmentation and not get_recursion_guard():
            stacked, labels = self.augmentation.apply(stacked, labels)

        # 优化：直接返回 uint8 张量，将除以 255.0 以及转 float 的算力推延到 GPU 端进行，节约 75% 的 PCIe 数据带宽
        image_tensor = torch.from_numpy(stacked)
        labels_tensor = torch.from_numpy(labels).float() if len(labels) > 0 else torch.zeros((0, 5),
                                                                                             dtype=torch.float32)
        return image_tensor, labels_tensor, idx

def yolo_collate_fn_with_indices(
        batch: list[tuple[torch.Tensor, torch.Tensor, int]]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    images, targets, sample_indices = zip(*batch)
    images_tensor = torch.stack(images, dim=0)

    all_targets = []
    for img_idx, img_targets in enumerate(targets):
        if len(img_targets) > 0:
            indices = torch.full((len(img_targets), 1), img_idx, dtype=img_targets.dtype)
            all_targets.append(torch.cat([indices, img_targets], dim=1))

    targets_tensor = torch.cat(all_targets, dim=0) if all_targets else torch.zeros((0, 6), dtype=torch.float32)
    sample_indices_tensor = torch.tensor(sample_indices, dtype=torch.long)
    return images_tensor, targets_tensor, sample_indices_tensor
