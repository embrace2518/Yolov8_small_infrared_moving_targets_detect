from __future__ import annotations

import json
import logging
import random
import re
import threading
from collections import OrderedDict
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path
from queue import Empty, Full, Queue
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
    augment: bool
    target_size: tuple[int, int]

    cache_images: bool = True
    cache_labels: bool = True
    prefetch: bool = True
    image_cache_size: int = 2048
    label_cache_size: int = 4096
    max_labels: int = 300
    prefetch_workers: int = 2
    prefetch_size: int = 128

    flip_prob: float = 0.5
    rotate_prob: float = 0.3
    rotate_degrees: int = 10
    brightness_jitter: float = 0.2
    contrast_jitter: float = 0.2

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
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    last_processed_frame: int = -1

    def __len__(self):
        return len(self.frame_paths)

class EnhancedYOLODataset(Dataset):
    def __init__(self, config: DatasetConfig, mode: str):
        self.config = config
        self.mode = mode
        self.do_augment = mode == "train" and config.augment

        self.image_dirs = self._normalize_dirs(config.images_dir)
        self.label_dirs = self._normalize_dirs(config.labels_dir)
        self.target_size = config.target_size

        self.flip_prob = config.flip_prob
        self.rotate_prob = config.rotate_prob
        self.rotate_degrees = config.rotate_degrees
        self.brightness_jitter = config.brightness_jitter
        self.contrast_jitter = config.contrast_jitter

        self.prefetch_enabled = config.prefetch

        self.logger = logging.getLogger(__name__)

        self.sequences = self._load_sequences()
        if not self.sequences:
            raise FileNotFoundError(f"no image files found at {config.images_dir}")

        self.samples = self._build_sample_index()
        self._validate_annotations()
        self._init_preprocessors()

        self._label_json_index: dict[tuple[str | int, str | int], dict] = {}
        if config.true_labels_json_path and Path(config.true_labels_json_path).exists():
            self._load_and_index_true_labels_json(self, config.true_labels_json_path)

        self._prefetch_cache_lock = threading.Lock()
        self._prefetch_cache: OrderedDict[int, tuple[np.ndarray, np.ndarray]] = OrderedDict()
        self._prefetch_queue: Queue[int | None] = Queue(maxsize=max(64, self.config.prefetch_size * 2))
        self._stop_prefetch = threading.Event()
        self._prefetch_threads: list[threading.Thread] = []
        self._read_fail_count = 0
        self._read_fail_log_limit = 20
        if self.config.prefetch:
            self._start_prefetch_threads()

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

        # 兼容旧目录：labels根下同名文件。
        for label_dir in self.label_dirs:
            candidate = label_dir / f"{image_path.stem}.txt"
            if candidate.exists():
                return candidate

        # ========== 新增第4步：从JSON索引按需生成 ==========
        # 如果以上都找不到，且我们有关联的JSON索引，则尝试从JSON生成
        if self._label_json_index:
            parent_name = image_path.parent.name
            stem_name = image_path.stem
            seq_match = re.search(r"(\d+)", parent_name)
            frame_match = re.search(r"(\d+)", stem_name)
            sequence_id = _to_int_or_self(seq_match.group(1)) if seq_match else parent_name
            frame = _to_int_or_self(frame_match.group(1)) if frame_match else stem_name
            lookup_key = (sequence_id, frame)
            json_item = self._label_json_index.get(lookup_key)
            if json_item is not None:
                target_label_path = same_dir_label
                target_label_path.parent.mkdir(parents=True, exist_ok=True)

                success = _generate_single_label_from_json(
                    self,
                    json_item=json_item,
                    target_label_path=target_label_path,
                    image_width = self.config.target_size[0],
                    image_height=self.config.target_size[1],
                    class_id=self.config.json_label_class_id,
                    fallback_box_width=self.config.json_fallback_box_width,
                    fallback_box_height=self.config.json_fallback_box_height,
                )
                if success:
                    return target_label_path
            # ===================================================
        # 4. 所有尝试都失败，返回一个（可能不存在的）占位路径
        # （注意：原代码的“3. 兼容旧目录”步骤现在变成了第3步，这是第4步）
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
            img_path.name
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

        # 预先构建CLAHE和gamma LUT，避免每帧重复创建对象。
        self._clahe = cv2.createCLAHE(
            clipLimit=self.clahe_params["clip_limit"],
            tileGridSize=self.clahe_params["tile_grid_size"],
        )
        gamma = max(float(self.clahe_params["gamma"]), 1e-6)
        self._gamma_lut = np.array([np.clip(((i / 255.0) ** gamma) * 255.0, 0, 255) for i in range(256)], dtype=np.uint8)

    def _apply_preprocessing(self, gray_image: np.ndarray, seq: SequenceData, frame_idx: int) -> np.ndarray:
        # A策略：只对顺序帧维持NUC状态，随机跳帧时重置该序列NUC。
        with seq.lock:
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
        enhanced = cv2.LUT(self._clahe.apply(denoised), self._gamma_lut)
        return enhanced

    def _cache_image(self, seq_idx: int, frame_idx: int, image: np.ndarray) -> None:
        if not self.config.cache_images:
            return
        seq = self.sequences[seq_idx]
        with seq.lock:
            seq.image_cache[frame_idx] = image.copy()

    def _load_image(self, seq_idx: int, frame_idx: int) -> np.ndarray:
        seq = self.sequences[seq_idx]
        if self.config.cache_images:
            with seq.lock:
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
        with seq.lock:
            seq.label_cache[frame_idx] = labels.copy()

    def _load_yolo_labels(self, seq_idx: int, frame_idx: int) -> np.ndarray:
        seq = self.sequences[seq_idx]
        if self.config.cache_labels:
            with seq.lock:
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
                if len(parts) != 5:
                    continue
                try:
                    class_id, x_center, y_center, width, height = (float(v) for v in parts)
                except ValueError:
                    continue

                if 0 <= x_center <= 1 and 0 <= y_center <= 1 and 0 <= width <= 1 and 0 <= height <= 1:
                    labels.append([class_id, x_center, y_center, width, height])

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

    def _start_prefetch_threads(self) -> None:
        worker_count = max(1, int(self.config.prefetch_workers))
        for i in range(worker_count):
            t = threading.Thread(target=self._prefetch_worker, name=f"prefetch-{i}", daemon=True)
            t.start()
            self._prefetch_threads.append(t)

    def _schedule_prefetch(self, idx: int) -> None:
        if not self.config.prefetch:
            return

        upper = min(len(self.samples), idx + 1 + int(self.config.prefetch_size))
        for candidate in range(idx + 1, upper):
            with self._prefetch_cache_lock:
                if candidate in self._prefetch_cache:
                    continue
            try:
                self._prefetch_queue.put_nowait(candidate)
            except Full:
                break

    def _prefetch_worker(self) -> None:
        while not self._stop_prefetch.is_set():
            try:
                idx = self._prefetch_queue.get(timeout=0.2)
            except Empty:
                continue

            if idx is None:
                self._prefetch_queue.task_done()
                break

            with self._prefetch_cache_lock:
                if idx in self._prefetch_cache:
                    self._prefetch_cache.move_to_end(idx)
                    self._prefetch_queue.task_done()
                    continue

            sample = self._load_sample_raw(idx)
            with self._prefetch_cache_lock:
                self._prefetch_cache[idx] = sample
                self._prefetch_cache.move_to_end(idx)
                while len(self._prefetch_cache) > max(1, int(self.config.prefetch_size)):
                    self._prefetch_cache.popitem(last=False)

            self._prefetch_queue.task_done()

    def _apply_augmentations(self, image: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if not self.do_augment:
            return image, labels

        augmented_image = image.copy()
        augmented_labels = labels.copy()
        h, w = image.shape[:2]

        if random.random() < self.flip_prob:
            augmented_image = cv2.flip(augmented_image, 1)
            if len(augmented_labels) > 0:
                augmented_labels[:, 1] = 1.0 - augmented_labels[:, 1]

        if random.random() < self.rotate_prob:
            angle = random.uniform(-self.rotate_degrees, self.rotate_degrees)
            center = (w // 2, h // 2)
            rot_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            augmented_image = cv2.warpAffine(augmented_image, rot_matrix, (w, h), flags=cv2.INTER_LINEAR)

        if random.random() < 0.5:
            alpha = 1.0 + random.uniform(-self.contrast_jitter, self.contrast_jitter)
            beta = random.uniform(-255.0 * self.brightness_jitter, 255.0 * self.brightness_jitter)
            augmented_image = cv2.convertScaleAbs(augmented_image, alpha=alpha, beta=beta)

        return augmented_image, augmented_labels

    def _resize_image(self, image: np.ndarray) -> np.ndarray:
        target_h, target_w = self.target_size
        return cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        image: np.ndarray
        labels: np.ndarray

        if self.config.prefetch:
            with self._prefetch_cache_lock:
                cached = self._prefetch_cache.pop(idx, None)
            if cached is not None:
                image, labels = cached
            else:
                image, labels = self._load_sample_raw(idx)
            self._schedule_prefetch(idx)
        else:
            image, labels = self._load_sample_raw(idx)

        image, labels = self._apply_augmentations(image, labels)
        image = self._resize_image(image)

        image_tensor = torch.from_numpy(image).float().unsqueeze(0) / 255.0
        labels_tensor = torch.from_numpy(labels).float() if len(labels) > 0 else torch.zeros((0, 5), dtype=torch.float32)
        return image_tensor, labels_tensor

    def get_image_with_boxes(self, idx: int) -> np.ndarray:
        image_tensor, labels_tensor = self[idx]
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
        if not self.prefetch:
            return
        self._stop_prefetch.set()
        for _ in self._prefetch_threads:
            try:
                self._prefetch_queue.put_nowait(None)
            except Full:
                pass
        for t in self._prefetch_threads:
            if t.is_alive():
                t.join(timeout=0.5)

    def __del__(self) -> None:
        self.close()


def yolo_collate_fn_with_indices(batch: list[tuple[torch.Tensor, torch.Tensor]]) -> tuple[torch.Tensor, torch.Tensor]:
    images, targets = zip(*batch)
    images_tensor = torch.stack(images, dim=0)

    all_targets = []
    for img_idx, img_targets in enumerate(targets):
        if len(img_targets) > 0:
            indices = torch.full((len(img_targets), 1), img_idx, dtype=img_targets.dtype)
            all_targets.append(torch.cat([indices, img_targets], dim=1))

    targets_tensor = torch.cat(all_targets, dim=0) if all_targets else torch.zeros((0, 6), dtype=torch.float32)
    return images_tensor, targets_tensor


def _to_int_or_self(value: str) -> int | str:
    try:
        return int(value)
    except ValueError:
        return value


def model_validate(predictions: list[Any], output_json: str | Path = "my_anno.json") -> list[dict[str, Any]]:
    """将 YOLO predict 结果转换为验证 JSON。"""
    validation_data: list[dict[str, Any]] = []

    for result in predictions:
        img_path = Path(str(result.path))
        parent_name = img_path.parent.name
        stem_name = img_path.stem

        seq_match = re.search(r"(\d+)", parent_name)
        frame_match = re.search(r"(\d+)", stem_name)
        sequence_id = _to_int_or_self(seq_match.group(1)) if seq_match else parent_name
        frame = _to_int_or_self(frame_match.group(1)) if frame_match else stem_name

        object_coords: list[list[float]] = []
        boxes = result.boxes
        if boxes is not None:
            for box in boxes.xyxy:
                x1, y1, x2, y2 = box.tolist()
                x_center = (x1 + x2) / 2.0
                y_center = (y1 + y2) / 2.0
                object_coords.append([x_center, y_center])

        validation_data.append(
            {
                "sequence_id": sequence_id,
                "frame": frame,
                "num_objects": len(object_coords),
                "object_coords": object_coords,
            }
        )

    output_path = Path(output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(validation_data, f, indent=4, ensure_ascii=False)

    print(f"validation json saved: {output_path}")
    return validation_data


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


def run_validate_json(
    model_path: str | Path,
    source: str | Path,
    output_json: str | Path = "my_anno.json",
    conf: float = 0.1,
    save: bool = False,
    project: str | Path = "runs/detect",
    name: str = "predict",
) -> list[dict[str, Any]]:
    """A 模式：直接执行 YOLO.predict 并生成验证 JSON。"""
    from ultralytics import YOLO

    model = YOLO(str(model_path))
    predictions = model.predict(
        source=str(source),
        conf=conf,
        save=save,
        project=str(project),
        name=name,
        exist_ok=True,
    )
    return model_validate(predictions, output_json=output_json)
