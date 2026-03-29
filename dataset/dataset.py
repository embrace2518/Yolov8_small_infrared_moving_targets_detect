import torch
from torch.utils.data import Dataset
import numpy as np
import cv2
from pathlib import Path
from typing import Tuple, List
import random
from dataclasses import dataclass
try:
    from .preprocess import PipelineConfig, SceneBasedNUC, read_gray_image, denoise_frame, apply_clahe_and_gamma
except ImportError:  # pragma: no cover
    from preprocess import PipelineConfig, SceneBasedNUC, read_gray_image, denoise_frame, apply_clahe_and_gamma


@dataclass
class DatasetConfig:
    # 路径配置
    images_dir: Path
    labels_dir: Path

    # 预处理配置
    preprocess_config: PipelineConfig

    # 增强配置
    augment: bool = True
    flip_prob: float = 0.5
    rotate_prob: float = 0.3
    rotate_degrees: int = 10
    brightness_jitter: float = 0.2
    contrast_jitter: float = 0.2

    # 其他配置
    target_size: Tuple[int, int] = (640, 640)  # YOLO常用尺寸
    cache_images: bool = True  # 是否缓存图片到内存
    max_labels: int = 100  # 每张图最大标签数


class EnhancedYOLODataset(Dataset):
    def __init__(self, config: DatasetConfig, mode: str = "train"):
        """
        Args:
            config: 数据集配置
            mode: "train" 或 "val"
        """
        self.config = config
        self.mode = mode
        self.images_dir = Path(config.images_dir)
        self.labels_dir = Path(config.labels_dir)
        self.target_size = config.target_size

        if mode == "train":
            self.do_augment = config.augment
        else:
            self.do_augment = False

        self.image_files = self._load_image_files()

        if len(self.image_files) == 0:
            raise FileNotFoundError(f"在 {self.images_dir} 中没有找到图片文件")

        # 检查对应的标注文件
        self._validate_annotations()

        # 初始化预处理模块
        self._init_preprocessors()

        # 图片缓存
        self.image_cache = {} if config.cache_images else None

        print(f"初始化 {mode} 数据集: {len(self.image_files)} 张图片")

    def _load_image_files(self) -> List[Path]:
        """加载所有图片文件"""
        image_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
        image_files = []

        for ext in image_exts:
            image_files.extend(self.images_dir.glob(f"*{ext}"))
            image_files.extend(self.images_dir.glob(f"*{ext.upper()}"))

        # 按文件名排序确保一致性
        return sorted(image_files, key=lambda x: x.stem)

    def _validate_annotations(self):
        """验证标注文件是否存在"""
        missing_labels = []

        for img_path in self.image_files:
            label_path = self.labels_dir / f"{img_path.stem}.txt"
            if not label_path.exists():
                missing_labels.append(img_path.name)

        if missing_labels:
            print(f"警告: 缺少 {len(missing_labels)} 个标注文件")
            if len(missing_labels) <= 5:
                for name in missing_labels[:5]:
                    print(f"  - {name}")

    def _init_preprocessors(self):
        """初始化预处理模块"""
        config = self.config.preprocess_config

        # 初始化NUC（每张图片独立）
        # 注意：在训练模式下，我们可能需要重置NUC的reference
        self.nuc_alpha = config.nuc_alpha

        # 存储其他预处理参数
        self.denoise_params = {
            'method': config.denoise_method,
            'kernel': config.denoise_kernel,
            'h': config.denoise_h
        }

        self.clahe_params = {
            'clip_limit': config.clahe_clip_limit,
            'tile_grid_size': config.clahe_tile_grid_size,
            'gamma': config.gamma
        }

    def _apply_preprocessing(self, gray_image: np.ndarray) -> np.ndarray:
        """
        应用完整的预处理流水线
        注意：这里NUC每次重新初始化，确保训练/推理一致性
        """
        # 1. NUC校正
        nuc = SceneBasedNUC(alpha=self.nuc_alpha)
        corrected = nuc.apply(gray_image)

        # 2. 去噪
        denoised = denoise_frame(
            corrected,
            self.denoise_params['method'],
            self.denoise_params['kernel'],
            self.denoise_params['h']
        )

        # 3. CLAHE增强 + Gamma校正
        enhanced = apply_clahe_and_gamma(
            denoised,
            clip_limit=self.clahe_params['clip_limit'],
            tile_grid_size=self.clahe_params['tile_grid_size'],
            gamma=self.clahe_params['gamma']
        )

        return enhanced

    def _load_image(self, img_path: Path) -> np.ndarray:
        """加载并预处理图片"""
        # 检查缓存
        if self.image_cache is not None and img_path in self.image_cache:
            return self.image_cache[img_path].copy()

        # 读取图片（使用你的read_gray_image函数）
        gray = read_gray_image(img_path)

        # 应用预处理
        processed = self._apply_preprocessing(gray)

        # 缓存
        if self.image_cache is not None:
            self.image_cache[img_path] = processed.copy()

        return processed

    def _load_yolo_labels(self, label_path: Path, img_shape: Tuple[int, int]) -> np.ndarray:
        """
        加载YOLO格式标注
        返回: [N, 5] 数组，每行: [class_id, x_center, y_center, width, height]
        坐标已归一化到 [0, 1]
        """
        if not label_path.exists():
            return np.zeros((0, 5), dtype=np.float32)

        labels = []
        h, w = img_shape[:2]

        with open(label_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                parts = line.split()
                if len(parts) != 5:
                    continue

                try:
                    class_id = float(parts[0])
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])

                    # 验证坐标范围
                    if (0 <= x_center <= 1 and 0 <= y_center <= 1 and
                            0 <= width <= 1 and 0 <= height <= 1):
                        labels.append([class_id, x_center, y_center, width, height])

                except ValueError:
                    continue

        if len(labels) > self.config.max_labels:
            labels = labels[:self.config.max_labels]

        return np.array(labels, dtype=np.float32) if labels else np.zeros((0, 5), dtype=np.float32)

    def _apply_augmentations(self, image: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """应用数据增强"""
        if not self.do_augment or len(labels) == 0:
            return image, labels

        h, w = image.shape[:2]
        augmented_image = image.copy()
        augmented_labels = labels.copy()

        # 1. 随机水平翻转
        if random.random() < self.config.flip_prob:
            augmented_image = cv2.flip(augmented_image, 1)
            if len(augmented_labels) > 0:
                # 翻转边界框：x_center = 1 - x_center
                augmented_labels[:, 1] = 1.0 - augmented_labels[:, 1]

        # 2. 随机旋转
        if random.random() < self.config.rotate_prob and len(augmented_labels) > 0:
            angle = random.uniform(-self.config.rotate_degrees, self.config.rotate_degrees)
            center = (w // 2, h // 2)
            rot_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            augmented_image = cv2.warpAffine(augmented_image, rot_matrix, (w, h))

            # 旋转边界框（简化处理，实际需要更复杂的坐标变换）
            # 这里简单实现，对于小角度旋转可以接受
            # 对于复杂场景，建议使用albumentations库

        # 3. 亮度/对比度调整
        if random.random() < 0.5:
            # 随机亮度
            alpha = 1.0 + random.uniform(-self.config.brightness_jitter, self.config.brightness_jitter)
            beta = random.uniform(-20, 20)
            augmented_image = cv2.convertScaleAbs(augmented_image, alpha=alpha, beta=beta)

        return augmented_image, augmented_labels

    def _resize_image_and_labels(self, image: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """调整图片和标签到目标尺寸"""
        target_h, target_w = self.target_size

        # 调整图片大小
        resized_image = cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

        # labels 是 YOLO 归一化坐标，resize 后仍然有效，不应再次按像素比例缩放。
        resized_labels = labels.copy() if len(labels) > 0 else labels

        return resized_image, resized_labels

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        获取单个样本

        Returns:
            image_tensor: [1, H, W] 灰度图tensor，值范围[0, 1]
            labels_tensor: [N, 5] 标签tensor，格式[class_id, x_center, y_center, width, height]
        """
        # 1. 加载图片
        img_path = self.image_files[idx]
        image = self._load_image(img_path)  # 已预处理

        # 2. 加载标注
        label_path = self.labels_dir / f"{img_path.stem}.txt"
        labels = self._load_yolo_labels(label_path, image.shape)

        # 3. 数据增强（仅训练模式）
        if self.do_augment:
            image, labels = self._apply_augmentations(image, labels)

        # 4. 调整到目标尺寸
        image, labels = self._resize_image_and_labels(image, labels)

        # 5. 转换为tensor
        # 图片: 归一化到[0, 1]，添加通道维度
        image_tensor = torch.from_numpy(image).float() / 255.0
        image_tensor = image_tensor.unsqueeze(0)  # [H, W] -> [1, H, W]

        # 标签: 转为tensor
        if len(labels) > 0:
            labels_tensor = torch.from_numpy(labels).float()
        else:
            labels_tensor = torch.zeros((0, 5), dtype=torch.float32)

        return image_tensor, labels_tensor

    def get_image_with_boxes(self, idx: int) -> np.ndarray:
        """获取带边界框的可视化图片（用于调试）"""
        image_tensor, labels_tensor = self[idx]
        image = (image_tensor.squeeze(0).numpy() * 255).astype(np.uint8)

        # 绘制边界框
        if len(labels_tensor) > 0:
            h, w = image.shape
            for label in labels_tensor:
                class_id, x_center, y_center, width, height = label

                # 转换为像素坐标
                x1 = int((x_center - width / 2) * w)
                y1 = int((y_center - height / 2) * h)
                x2 = int((x_center + width / 2) * w)
                y2 = int((y_center + height / 2) * h)

                # 绘制矩形
                cv2.rectangle(image, (x1, y1), (x2, y2), (255, 255, 255), 2)
                cv2.putText(image, f"{int(class_id)}", (x1, max(y1 - 5, 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return image


def yolo_collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    """
    YOLO数据集的collate函数
    处理变长的标签

    Args:
        batch: 列表，每个元素是(image, labels)

    Returns:
        images: [B, 1, H, W]
        targets: 列表，每个元素是[N_i, 5]的标签
    """
    images, targets = zip(*batch)

    # 堆叠图片
    images = torch.stack(images, dim=0)  # [B, 1, H, W]

    # 标签保持为列表（因为长度不同）
    return images, list(targets)


def yolo_collate_fn_with_indices(batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    YOLO数据集的collate函数（带索引版本）
    将所有标签合并为一个tensor，第一列为图片索引

    Returns:
        images: [B, 1, H, W]
        targets: [总目标数, 6] 格式: [img_idx, class_id, x_center, y_center, width, height]
    """
    images, targets = zip(*batch)

    # 堆叠图片
    images = torch.stack(images, dim=0)  # [B, 1, H, W]

    # 合并所有标签，添加图片索引
    all_targets = []
    for img_idx, img_targets in enumerate(targets):
        if len(img_targets) > 0:
            # 添加图片索引作为第一列
            indices = torch.full((len(img_targets), 1), img_idx)
            targets_with_idx = torch.cat([indices, img_targets], dim=1)
            all_targets.append(targets_with_idx)

    if all_targets:
        all_targets = torch.cat(all_targets, dim=0)  # [总目标数, 6]
    else:
        all_targets = torch.zeros((0, 6), dtype=torch.float32)

    return images, all_targets
