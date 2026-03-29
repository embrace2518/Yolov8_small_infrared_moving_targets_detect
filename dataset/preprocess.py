from argparse import ArgumentParser, Namespace
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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


class SceneBasedNUC:
    """A simple scene-based non-uniformity correction estimator.

    It keeps a low-frequency reference field with exponential moving average and
    subtracts the estimated fixed-pattern component from each frame.
    """

    def __init__(self, alpha: float):
        self.alpha = float(alpha)
        self.reference: np.ndarray | None = None

    def apply(self, gray: np.ndarray) -> np.ndarray:
        frame = gray.astype(np.float32)
        low_freq = cv2.GaussianBlur(frame, (0, 0), sigmaX=15, sigmaY=15)
        if self.reference is None:
            self.reference = low_freq.copy()
        else:
            self.reference = (1.0 - self.alpha) * self.reference + self.alpha * low_freq

        corrected = frame - self.reference + float(np.mean(self.reference))
        return np.clip(corrected, 0, 255).astype(np.uint8)


def load_config(config_path: Path) -> PipelineConfig:
    with config_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    io_cfg = raw.get("io", {})
    nuc_cfg = raw.get("nuc", {})
    denoise_cfg = raw.get("denoise", {})
    clahe_cfg = raw.get("clahe", {})
    image_exts = tuple(e.lower() for e in io_cfg.get("image_exts", list(IMAGE_EXTS)))
    tile_grid_raw = clahe_cfg.get("tile_grid_size", [8, 8])
    tile_grid_size = (int(tile_grid_raw[0]), int(tile_grid_raw[1]))

    return PipelineConfig(
        image_exts=image_exts,
        nuc_alpha=float(nuc_cfg.get("alpha", 0.02)),
        denoise_method=str(denoise_cfg.get("method", "gaussian")).lower(),
        denoise_kernel=int(denoise_cfg.get("kernel", 3)),
        denoise_h=int(denoise_cfg.get("h", 10)),
        clahe_clip_limit=float(clahe_cfg.get("clip_limit", 2.5)),
        clahe_tile_grid_size=tile_grid_size,
        gamma=float(clahe_cfg.get("gamma", 0.95)),
    )


def read_gray_image(path: Path) -> np.ndarray:
    if not path.exists():
        raise ImageReadError(path, "path_not_found")

    # Direct grayscale load avoids cvtColor channel mismatch on single-channel images.
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        # Windows-safe fallback for paths/content that imread cannot decode directly.
        try:
            raw = np.fromfile(str(path), dtype=np.uint8)
            if raw.size > 0:
                image = cv2.imdecode(raw, cv2.IMREAD_GRAYSCALE)
        except Exception:
            image = None

    if image is None:
        raise ImageReadError(path, "decode_failed")

    if image.ndim == 3:
        # Defensive fallback for rare decoders returning HxWx1/HxWx3/HxWx4.
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

    if image.dtype != np.uint8:
        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return image


def denoise_frame(gray: np.ndarray, method: str, kernel: int, h: int) -> np.ndarray:
    kernel = max(1, int(kernel)) if kernel % 2 == 1 else kernel + 1
    if method == "median":
        return cv2.medianBlur(gray, kernel)
    if method == "nlm":
        return cv2.fastNlMeansDenoising(gray, None, h=h, templateWindowSize=7, searchWindowSize=21)
    return cv2.GaussianBlur(gray, (kernel, kernel), 0)


def apply_clahe_and_gamma(gray: np.ndarray, clip_limit: float, tile_grid_size: tuple[int, int], gamma: float) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    enhanced = clahe.apply(gray)
    gamma = max(gamma, 1e-6)
    lut = np.array([np.clip(((i / 255.0) ** gamma) * 255.0, 0, 255) for i in range(256)], dtype=np.uint8)
    return cv2.LUT(enhanced, lut)

