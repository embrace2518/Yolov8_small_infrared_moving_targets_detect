from __future__ import annotations

import json
import logging
import re
from collections import OrderedDict
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from .preprocess import ImageReadError, PipelineConfig, SceneBasedNUC, denoise_frame, read_gray_image


@dataclass
class DatasetConfig:
    images_dir: Path | str | list[Path | str]
    labels_dir: Path | str | list[Path | str]
    preprocess_config: PipelineConfig
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

    def __len__(self):
        return len(self.frame_paths)

class YOLODataset(Dataset):
    def __init__(self, config: DatasetConfig, mode: str):
        self.config = config
        self.mode = mode
        self.image_dirs = self._normalize_dirs(config.images_dir)
        self.label_dirs = self._normalize_dirs(config.labels_dir)
        self.target_size = config.target_size
        self.prefetch_enabled = config.prefetch
        self.logger = logging.getLogger(__name__)
        self._label_json_index: dict[tuple[str | int, str | int], dict] = {}

        if config.true_labels_json_path and Path(config.true_labels_json_path).exists():
            self._load_and_index_true_labels_json(self, config.true_labels_json_path)

        self.sequences = self._load_sequences()
        if not self.sequences:
            raise FileNotFoundError(f"no image files found at {config.images_dir}")

        self.samples = self._build_sample_index()
        self._validate_annotations()
        self._init_preprocessors()

        self._read_fail_count = 0
        self._read_fail_log_limit = 20

    @staticmethod
    def _load_and_index_true_labels_json(self, json_path: Path | str) -> None:
        with Path(json_path).open('r', encoding='utf-8') as f:
            data = json.load(f)
        index = {}
        for item in data:
            seq_id_raw = item.get("sequence_id")
            frame_raw = item.get("frame")
            if seq_id_raw is None or frame_raw is None:
                continue
            seq_id = _to_int_or_self(str(seq_id_raw))
            frame = _to_int_or_self(str(frame_raw))
            index[(seq_id, frame)] = item
        self._label_json_index = index

    @staticmethod
    def _normalize_dirs(path_like: Path | str | list[Path | str]) -> list[Path]:
        if isinstance(path_like, (str, Path)):
            return [Path(path_like)]
        return [Path(p) for p in path_like]

    @staticmethod
    def _image_sort_key(path: Path) -> tuple[str, int, str]:
        stem = path.stem
        numbers = list(map(int, re.findall(r'\d+', stem)))
        primary_num = numbers[-1] if numbers else -1
        return str(path.parent), primary_num, stem

    def _load_image_files(self) -> list[Path]:
        image_exts = self.config.preprocess_config.image_exts
        image_files: list[Path] = []
        for img_dir in self.image_dirs:
            for ext in image_exts:
                image_files.extend(img_dir.rglob(f"*{ext}"))
                image_files.extend(img_dir.rglob(f"*{ext.upper()}"))
        return sorted(image_files, key=self._image_sort_key)

    def _load_sequences(self) -> list[SequenceData]:
        grouped: dict[str, list[Path]] = {}
        for img_path in self._load_image_files():
            grouped.setdefault(str(img_path.parent), []).append(img_path)

        sequences: list[SequenceData] = []
        for parent_str, frame_paths in grouped.items():
            ordered_frames = sorted(frame_paths, key=self._image_sort_key)
            label_paths = [self._resolve_label_path(p) for p in ordered_frames]
            sequences.append(SequenceData(name=Path(parent_str).name, frame_paths=ordered_frames, label_paths=label_paths,))
        sequences.sort(key=lambda s: s.name)
        return sequences

    def _relative_to_image_root(self, image_path: Path) -> Path:
        for root in self.image_dirs:
            with suppress(ValueError):
                return image_path.relative_to(root)
        return Path(image_path.name)

    def _resolve_label_path(self, image_path: Path) -> Path:
        same_dir_label = image_path.with_suffix(".txt")
        if same_dir_label.exists():
            return same_dir_label

        # 次优先：labels_dir下保留相对路径。
        rel = self._relative_to_image_root(image_path).with_suffix(".txt")
        for label_dir in self.label_dirs:
            candidate = label_dir / rel
            if candidate.exists():
                return candidate

        # 从JSON索引按需生成
        if self._label_json_index:
            parent_name = image_path.parent.name
            stem_name = image_path.stem
            seq_match = re.search(r"(\d+)", parent_name)
            frame_match = re.search(r"(\d+)", stem_name)
            sequence_id = _to_int_or_self(seq_match.group(1)) if seq_match else parent_name
            frame = _to_int_or_self(frame_match.group(1)) if frame_match else stem_name

            json_item = self._label_json_index.get((sequence_id, frame))
            if json_item is not None:
                same_dir_label.parent.mkdir(parents=True, exist_ok=True)
                if _generate_single_label_from_json(
                    self,
                    json_item=json_item,
                    target_label_path=same_dir_label,
                    image_width=self.config.target_size[0],
                    image_height=self.config.target_size[1],
                    class_id=self.config.json_label_class_id,
                    fallback_box_width=self.config.json_fallback_box_width,
                    fallback_box_height=self.config.json_fallback_box_height,
                ):
                    return same_dir_label

        return same_dir_label

    def _build_sample_index(self) -> list[tuple[int, int]]:
        """构建数据集索引列表，每个元素为 (序列索引, 帧索引)。"""
        return [
            (seq_idx, frame_idx)
            for seq_idx, seq in enumerate(self.sequences)
            for frame_idx in range(len(seq))  # 使用 seq.__len__() 接口
        ]

    def _validate_annotations(self) -> None:
        """验证所有图片是否都有对应的标签文件，并报告缺失项。"""
        missing_labels = [
            (img_path.name, str(label_path))
            for seq in self.sequences
            for img_path, label_path in zip(seq.frame_paths, seq.label_paths)
            if not label_path.exists()
        ]
        if missing_labels:
            self.logger.warning("发现 %d 个图片缺失对应的标签文件。", len(missing_labels))
            for name, full_path in missing_labels[:5]:
                self.logger.warning("缺失标签: 文件 %s (路径: %s)", name, full_path)
            if len(missing_labels) > 5:
                self.logger.warning("... 以及另外 %d 个文件。", len(missing_labels) - 5)

    def _init_preprocessors(self) -> None:
        cfg = self.config.preprocess_config
        self.nuc_alpha = cfg.nuc_alpha
        self.denoise_params = {
            "method": cfg.denoise_method,
            "kernel": cfg.denoise_kernel,
            "h": cfg.denoise_h,
        }
        self.clahe_params = {
            "clip_limit": cfg.clahe_clip_limit,
            "tile_grid_size": cfg.clahe_tile_grid_size,
            "gamma": cfg.gamma,
        }

        # 预先构建 LUT
        gamma = max(float(self.clahe_params["gamma"]), 1e-6)
        self._gamma_lut = np.array([np.clip(((i / 255.0) ** gamma) * 255.0, 0, 255) for i in range(256)], dtype=np.uint8)

    def __getstate__(self):
        # 移除不可序列化的C++对象
        state = self.__dict__.copy()
        state.pop("_clahe", None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        # 恢复状态时，置空 CLAHE 这样每个子进程可自行创建
        self._clahe = None

        # 如果 _label_json_index 是空字典，不影响
        pass

    def _apply_preprocessing(self, gray_image: np.ndarray, seq: SequenceData, frame_idx: int) -> np.ndarray:
        # A策略：只对顺序帧维持NUC状态，随机跳帧时重置该序列NUC。
        if seq.nuc is None or frame_idx != seq.last_processed_frame + 1:
            seq.nuc = SceneBasedNUC(alpha=self.nuc_alpha)
            seq.last_processed_frame = -1
        corrected = seq.nuc.apply(gray_image)
        seq.last_processed_frame = frame_idx

        denoised = denoise_frame(
            corrected,
            self.denoise_params["method"],
            self.denoise_params["kernel"],
            self.denoise_params["h"],
        )
        
        if getattr(self, "_clahe", None) is None:
            self._clahe = cv2.createCLAHE(
                clipLimit=self.clahe_params["clip_limit"],
                tileGridSize=self.clahe_params["tile_grid_size"],
            )

        enhanced = cv2.LUT(self._clahe.apply(denoised), self._gamma_lut)
        return enhanced

    def _cache_image(self, seq_idx: int, frame_idx: int, image: np.ndarray) -> None:
        if not self.config.cache_images:
            return
        seq = self.sequences[seq_idx]
        seq.image_cache[frame_idx] = image.copy()
        if len(seq.image_cache) > 50:  # 限制单个序列的图片缓存数量
            seq.image_cache.popitem(last=False)

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
                print(f"[Dataset] read failed, use placeholder: {e}")
                if self._read_fail_count == self._read_fail_log_limit:
                    print("[Dataset] too many read errors, suppressing further logs...")

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
        if len(seq.label_cache) > 200: # 限制标签缓存数量
            seq.label_cache.popitem(last=False)

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

    def _load_sample_raw(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        seq_idx, frame_idx = self.samples[idx]
        image = self._load_image(seq_idx, frame_idx)
        labels = self._load_yolo_labels(seq_idx, frame_idx)
        return image, labels

    def _resize_image(self, image: np.ndarray) -> np.ndarray:
        target_h, target_w = self.target_size
        return cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        image, labels = self._load_sample_raw(idx)
        
        # 时序堆叠：如果配置了3通道输入，我们将 t-1, t, t+1 堆叠起来
        channels = int(getattr(self, "channels", 1))
        
        if channels == 3:
            seq_idx, frame_idx = self.samples[idx]
            seq_len = len(self.sequences[seq_idx])
            
            # 获取前后帧 (边界处理：超出边界则拿当前帧补)
            prev_frame_idx = max(0, frame_idx - 1)
            next_frame_idx = min(seq_len - 1, frame_idx + 1)
            
            image_prev = self._load_image(seq_idx, prev_frame_idx)
            image_next = self._load_image(seq_idx, next_frame_idx)
            
            # 缩放
            image = self._resize_image(image)
            image_prev = self._resize_image(image_prev)
            image_next = self._resize_image(image_next)
            
            # 堆叠成 HWC -> CHW 会在最后处理，当前是 HxW 的灰度图
            stacked = np.stack([image_prev, image, image_next], axis=0) # Shape: 3 x H x W
            image_tensor = torch.from_numpy(stacked).float() / 255.0
        else:
            image = self._resize_image(image)
            image_tensor = torch.from_numpy(image).float().unsqueeze(0) / 255.0
        
        labels_tensor = torch.from_numpy(labels).float() if len(labels) > 0 else torch.zeros((0, 5), dtype=torch.float32)
        return image_tensor, labels_tensor, idx

    def get_yolo_sample_meta(self, idx: int, image_shape: tuple[int, int] | None = None) -> dict[str, Any]:
        """Return lightweight metadata required by Ultralytics train/val loops."""
        seq_idx, frame_idx = self.samples[idx]
        seq = self.sequences[seq_idx]
        target_h, target_w = image_shape or self.target_size
        target_h, target_w = int(target_h), int(target_w)
        return {
            "im_file": str(seq.frame_paths[frame_idx]),
            "ori_shape": (target_h, target_w),
            "resized_shape": (target_h, target_w),
            "ratio_pad": ((1.0, 1.0), (0.0, 0.0)),
        }

    def get_image_with_boxes(self, idx: int) -> np.ndarray:
        image_tensor, labels_tensor, _ = self[idx]
        if image_tensor.ndim == 3 and image_tensor.shape[0] == 3:
            image = (image_tensor[0].numpy() * 255).astype(np.uint8)
        else:
            image = (image_tensor.squeeze(0).numpy() * 255).astype(np.uint8)

        if len(labels_tensor) > 0:
            h, w = image.shape
            for label in labels_tensor:
                class_id, x_center, y_center, width, height = label
                x1 = int((x_center - width / 2) * w)
                y1 = int((y_center - height / 2) * h)
                x2 = int((x_center + width / 2) * w)
                y2 = int((y_center + height / 2) * h)
                cv2.rectangle(image, (x1, y1), (x2, y2), (255, 255, 255), 2)
                cv2.putText(image, f"{int(class_id)}", (x1, max(y1 - 5, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255, 1)

        return image

    def close(self) -> None:
        pass

    def __del__(self) -> None:
        self.close()


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

def _to_int_or_self(value: str) -> int | str:
    try:
        return int(value)
    except ValueError:
        return value

def _xyxy_to_xywh_norm(x1: float, y1: float, x2: float, y2: float, image_w: int, image_h: int) -> tuple[float, float, float, float]:
    bw = max(0.0, x2 - x1)
    bh = max(0.0, y2 - y1)
    xc = x1 + bw / 2.0
    yc = y1 + bh / 2.0
    return xc / image_w, yc / image_h, bw / image_w, bh / image_h


def _generate_single_label_from_json(self,
                                     json_item: dict,
                                     target_label_path: Path,
                                     image_width: int,
                                     image_height: int,
                                     class_id: int = 0,
                                     fallback_box_width: int = 16,
                                     fallback_box_height: int = 16) -> bool:
    """根据单个JSON条目生成YOLO标签文件。"""
    object_bboxes = json_item.get("object_bboxes")
    object_coords = json_item.get("object_coords", [])

    lines: list[str] = []
    if isinstance(object_bboxes, list) and len(object_bboxes) > 0:
        for box in object_bboxes:
            if not isinstance(box, list) or len(box) != 4:
                continue
            try:
                a, b, c, d = (float(v) for v in box)
            except (ValueError, TypeError):
                continue

            if c > a and d > b:
                x_norm, y_norm, w_norm, h_norm = _xyxy_to_xywh_norm(a, b, c, d, image_width, image_height)
            else:
                x_norm = a / image_width
                y_norm = b / image_height
                w_norm = c / image_width
                h_norm = d / image_height

            x_norm = float(np.clip(x_norm, 0.0, 1.0))
            y_norm = float(np.clip(y_norm, 0.0, 1.0))
            w_norm = float(np.clip(w_norm, 0.0, 1.0))
            h_norm = float(np.clip(h_norm, 0.0, 1.0))
            lines.append(f"{class_id} {x_norm:.6f} {y_norm:.6f} {w_norm:.6f} {h_norm:.6f}")
    elif object_coords:  # 回退逻辑：使用object_coords和固定框
        for coord in object_coords:
            if not isinstance(coord, list) or len(coord) != 2:
                continue
            try:
                x_center, y_center = float(coord[0]), float(coord[1])
            except (ValueError, TypeError):
                continue
            x_norm = float(np.clip(x_center / image_width, 0.0, 1.0))
            y_norm = float(np.clip(y_center / image_height, 0.0, 1.0))
            w_norm = float(np.clip(fallback_box_width / image_width, 0.0, 1.0))
            h_norm = float(np.clip(fallback_box_height / image_height, 0.0, 1.0))
            lines.append(f"{class_id} {x_norm:.6f} {y_norm:.6f} {w_norm:.6f} {h_norm:.6f}")

    try:
        with target_label_path.open("w", encoding="utf-8") as f:
            if lines:
                f.write("\n".join(lines) + "\n")
            # 注意：即使lines为空（即无目标），也创建一个空文件，以标记“此图片已处理，确实无目标”
            # 这可以防止下次运行时再次查找JSON。
        return True
    except IOError as e:
        print(f"[Dataset] 警告: 无法写入标签文件 {target_label_path}: {e}")
        return False

