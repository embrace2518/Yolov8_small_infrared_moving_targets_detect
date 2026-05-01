from argparse import ArgumentParser, Namespace
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import yaml

IMAGE_EXTS = {".png", ".jpg"}


class ImageReadError(ValueError):
    def __init__(self, path: Path, reason: str):
        self.path = path
        self.reason = reason
        super().__init__(f"Failed to read image: {path} (reason={reason})")


@dataclass
class PipelineConfig:
    image_exts: tuple[str, ...]
    nuc_alpha: float
    denoise_method: str
    denoise_kernel: int
    denoise_h: int
    clahe_clip_limit: float
    clahe_tile_grid_size: tuple[int, int]
    gamma: float
    # 16bit 拉伸参数
    stretch_low_pct: float = 1.0      # 低端截断百分位
    stretch_high_pct: float = 99.0    # 高端截断百分位


@dataclass
class UnifiedConfig:
    enable_preprocess: bool
    enable_augmentation: bool
    preprocess_config: PipelineConfig
    augmentation_config: dict


class SceneBasedNUC:
    """Scene-based non-uniformity correction for infrared imagery.

    Keeps a low-frequency reference field via exponential moving average and
    subtracts the estimated fixed-pattern from each frame.

    Note: NUC is designed for raw sensor data before any contrast enhancement.
    When applied after CLAHE/Gamma, the fixed-pattern has been amplified and
    may become unrecoverable. Always apply NUC as the first processing step.
    """

    def __init__(self, alpha: float):
        self.alpha = float(alpha)
        self.reference: np.ndarray | None = None

    def apply(self, gray: np.ndarray) -> np.ndarray:
        frame = gray.astype(np.float32)

        # 高斯模糊核大小自适应：至少取图像短边的 1/20
        h, w = frame.shape[:2]
        sigma = max(15, min(h, w) / 20)
        ksize = int(2 * round(sigma * 2) + 1)  # 确保奇数
        low_freq = cv2.GaussianBlur(frame, (0, 0), sigmaX=sigma, sigmaY=sigma)

        if self.reference is None:
            self.reference = low_freq.copy()
        else:
            self.reference = (1.0 - self.alpha) * self.reference + self.alpha * low_freq

        # 减去固定模式噪声，保留平均亮度
        corrected = frame - self.reference + float(np.mean(self.reference))
        return np.clip(corrected, 0, 255).astype(np.uint8)


def load_config(config_path: Path) -> UnifiedConfig:
    with config_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    io_cfg = raw.get("io", {})
    nuc_cfg = raw.get("nuc", {})
    denoise_cfg = raw.get("denoise", {})
    clahe_cfg = raw.get("clahe", {})
    stretch_cfg = raw.get("stretch", {})
    augmentation_cfg = raw.get("augmentation", {})
    image_exts = tuple(e.lower() for e in io_cfg.get("image_exts", list(IMAGE_EXTS)))
    tile_grid_raw = clahe_cfg.get("tile_grid_size", [8, 8])
    tile_grid_size = (int(tile_grid_raw[0]), int(tile_grid_raw[1]))

    preprocess_config = PipelineConfig(
        image_exts=image_exts,
        nuc_alpha=float(nuc_cfg.get("alpha", 0.02)),
        denoise_method=str(denoise_cfg.get("method", "gaussian")).lower(),
        denoise_kernel=int(denoise_cfg.get("kernel", 3)),
        denoise_h=int(denoise_cfg.get("h", 10)),
        clahe_clip_limit=float(clahe_cfg.get("clip_limit", 2.5)),
        clahe_tile_grid_size=tile_grid_size,
        gamma=float(clahe_cfg.get("gamma", 0.95)),
        stretch_low_pct=float(stretch_cfg.get("low_pct", 1.0)),
        stretch_high_pct=float(stretch_cfg.get("high_pct", 99.0)),
    )

    return UnifiedConfig(
        enable_preprocess=bool(raw.get("enable_preprocess", True)),
        enable_augmentation=bool(raw.get("enable_augmentation", True)),
        preprocess_config=preprocess_config,
        augmentation_config=augmentation_cfg,
    )


def read_gray_image(path: Path, stretch_low_pct: float = 1.0, stretch_high_pct: float = 99.0) -> np.ndarray:
    if not path.exists():
        raise ImageReadError(path, "path_not_found")

    # 先用 IMREAD_UNCHANGED 读取（保留16bit深度），再按需转换
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        # Windows-safe fallback
        try:
            raw = np.fromfile(str(path), dtype=np.uint8)
            if raw.size > 0:
                image = cv2.imdecode(raw, cv2.IMREAD_UNCHANGED)
        except Exception:
            image = None

    if image is None:
        raise ImageReadError(path, "decode_failed")

    # 处理多通道 -> 单通道灰度
    if image.ndim == 3:
        channels = image.shape[2]
        if channels == 1:
            image = image[:, :, 0]
        elif channels == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        elif channels == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
        else:
            raise ImageReadError(path, f"unsupported_channels_{channels}")

    if image.size == 0:
        raise ImageReadError(path, "empty_image")

    # 关键修复：如果图像是16bit深度（0-65535），用百分位拉伸到8bit
    if image.dtype == np.uint16:
        # 用百分位数做自适应拉伸，避免极值噪声影响
        low, high = np.percentile(image, [stretch_low_pct, stretch_high_pct])
        low = max(low, 0)
        high = min(high, 65535)
        if high > low:
            image = np.clip((image.astype(np.float32) - low) / (high - low) * 255.0, 0, 255).astype(np.uint8)
        else:
            image = np.zeros_like(image, dtype=np.uint8)
    elif image.dtype != np.uint8:
        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    return image


def denoise_frame(gray: np.ndarray, method: str, kernel: int, h: int) -> np.ndarray:
    kernel = max(1, int(kernel)) if kernel % 2 == 1 else kernel + 1
    if method == "median":
        return cv2.medianBlur(gray, kernel)
    if method == "nlm":
        return cv2.fastNlMeansDenoising(gray, None, h=h, templateWindowSize=7, searchWindowSize=21)
    if method == "bilateral":
        # 双边滤波：保边去噪，适合小目标
        # sigmaColor=10, sigmaSpace=kernel 对小目标友好
        return cv2.bilateralFilter(gray, d=kernel, sigmaColor=10, sigmaSpace=kernel)
    return cv2.GaussianBlur(gray, (kernel, kernel), 0)


def apply_clahe_and_gamma(gray: np.ndarray, clip_limit: float, tile_grid_size: tuple[int, int],
                          gamma: float) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    enhanced = clahe.apply(gray)
    gamma = max(gamma, 1e-6)
    lut = np.array([np.clip(((i / 255.0) ** gamma) * 255.0, 0, 255) for i in range(256)], dtype=np.uint8)
    return cv2.LUT(enhanced, lut)


class ImagePreprocessor:
    """图像预处理管道，封装所有预处理步骤"""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.nuc_alpha = config.nuc_alpha
        self.denoise_params = {
            "method": config.denoise_method,
            "kernel": config.denoise_kernel,
            "h": config.denoise_h,
        }
        self.clahe_clip_limit = float(config.clahe_clip_limit)
        self.clahe_tile_grid_size = tuple(config.clahe_tile_grid_size)
        self.clahe_gamma = float(config.gamma)

        # 预计算伽马校正LUT
        gamma = max(self.clahe_gamma, 1e-6)
        self._gamma_lut = np.array([np.clip(((i / 255.0) ** gamma) * 255.0, 0, 255) for i in range(256)],
                                   dtype=np.uint8)
        self._clahe = None  # 延迟创建CLAHE对象

    def get_gamma_lut(self) -> np.ndarray:
        """获取伽马校正查找表"""
        return self._gamma_lut

    def get_denoise_params(self) -> dict:
        """获取去噪参数"""
        return self.denoise_params

    def get_clahe_params(self) -> Tuple[float, Tuple[int, int]]:
        """获取CLAHE参数"""
        return self.clahe_clip_limit, self.clahe_tile_grid_size

    def get_nuc_alpha(self) -> float:
        """获取NUC alpha参数"""
        return self.nuc_alpha

    def process_image(self, gray_image: np.ndarray, seq_nuc: Optional[SceneBasedNUC] = None,
                      frame_idx: int = 0, last_processed_frame: int = -1) -> np.ndarray:
        """
        应用完整的图像预处理流程

        Args:
            gray_image: 输入灰度图像 [H, W] uint8
            seq_nuc: 序列的NUC校正器（可变对象，状态会在内部更新）
            frame_idx: 当前帧索引
            last_processed_frame: 上一帧处理索引

        Returns:
            处理后的图像 [H, W] uint8

        Note:
            seq_nuc 和 last_processed_frame 是 SequenceData 的属性，
            作为可变对象传入，NUC状态会在内部更新。
            但 last_processed_frame 需要在外部调用后手动更新！
        """
        # 判断是否需要重置NUC：跳帧或首次调用
        if seq_nuc is None or frame_idx != last_processed_frame + 1:
            # 重置NUC：直接修改传入对象的内部状态
            if seq_nuc is not None:
                seq_nuc.reference = None
            # 注意：last_processed_frame 需由调用方在返回后更新

        # ===== 预处理管道 =====
        # 顺序说明：
        #   1) 先去噪：去除传感器读出噪声，让NUC得到更纯净的背景估计
        #   2) NUC：减去固定模式噪声（基于多帧累积的低频背景）
        #   3) CLAHE：增强局部对比度（红外小目标往往对比度极低）
        #   4) Gamma：微调整体亮度响应曲线
        #   5) 注意：CLAHE 在 NUC 之后是必要的，
        #      因为 NUC 后的残差噪声幅度已降低，CLAHE 不会过度放大噪声

        denoised = denoise_frame(
            gray_image,
            str(self.denoise_params["method"]),
            int(self.denoise_params["kernel"]),
            int(self.denoise_params["h"]),
        )

        corrected = seq_nuc.apply(denoised) if seq_nuc is not None else denoised

        if self._clahe is None:
            self._clahe = cv2.createCLAHE(
                clipLimit=self.clahe_clip_limit,
                tileGridSize=self.clahe_tile_grid_size,
            )

        enhanced = cv2.LUT(self._clahe.apply(corrected), self._gamma_lut)
        return enhanced

    def resize_image(self, image: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        """调整图像大小，对小目标使用更安全的插值方式"""
        target_h, target_w = target_size
        h, w = image.shape[:2]
        # 如果是缩小：用 INTER_AREA 避免小目标混叠
        # 如果是放大：用 INTER_LINEAR 平滑
        if h > target_h or w > target_w:
            interp = cv2.INTER_AREA
        else:
            interp = cv2.INTER_LINEAR
        return cv2.resize(image, (target_w, target_h), interpolation=interp)

    def create_nuc_corrector(self) -> SceneBasedNUC:
        """创建NUC校正器"""
        return SceneBasedNUC(alpha=self.nuc_alpha)

    def __getstate__(self):
        """序列化状态，移除不可pickle的OpenCV对象"""
        state = self.__dict__.copy()
        state.pop("_clahe", None)
        return state

    def __setstate__(self, state):
        """反序列化状态"""
        self.__dict__.update(state)
        self._clahe = None
