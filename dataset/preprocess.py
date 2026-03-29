from argparse import ArgumentParser, Namespace
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

IMAGE_EXTS = {".png", ".jpg"}


@dataclass
class PipelineConfig:
    input_root: Path
    output_root: Path
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


def natural_key(path: Path) -> tuple[int, Any]:
    stem = path.stem
    return (0, int(stem)) if stem.isdigit() else (1, stem.lower())


def load_config(config_path: Path) -> PipelineConfig:
    with config_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    io_cfg = raw.get("io", {})
    nuc_cfg = raw.get("nuc", {})
    denoise_cfg = raw.get("denoise", {})
    clahe_cfg = raw.get("clahe", {})
    input_root = Path(io_cfg.get("input_root", "D:/Dataset/train")).resolve()
    output_root = Path(io_cfg.get("output_root", "D:/Dataset_preprocessed/train")).resolve()
    image_exts = tuple(e.lower() for e in io_cfg.get("image_exts", list(IMAGE_EXTS)))
    tile_grid_raw = clahe_cfg.get("tile_grid_size", [8, 8])
    tile_grid_size = (int(tile_grid_raw[0]), int(tile_grid_raw[1]))

    return PipelineConfig(
        input_root=input_root,
        output_root=output_root,
        image_exts=image_exts,
        nuc_alpha=float(nuc_cfg.get("alpha", 0.02)),
        denoise_method=str(denoise_cfg.get("method", "gaussian")).lower(),
        denoise_kernel=int(denoise_cfg.get("kernel", 3)),
        denoise_h=int(denoise_cfg.get("h", 10)),
        clahe_clip_limit=float(clahe_cfg.get("clip_limit", 2.5)),
        clahe_tile_grid_size=tile_grid_size,
        gamma=float(clahe_cfg.get("gamma", 0.95)),
    )


def list_frames(sequence_dir: Path, image_exts: tuple[str, ...]) -> list[Path]:
    frames = [p for p in sequence_dir.iterdir() if p.is_file() and p.suffix.lower() in image_exts]
    return sorted(frames, key=natural_key)


def ensure_odd(kernel: int) -> int:
    kernel = max(1, int(kernel))
    return kernel if kernel % 2 == 1 else kernel + 1


def read_gray_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Failed to read image: {path}")
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if image.dtype != np.uint8:
        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return image


def denoise_frame(gray: np.ndarray, method: str, kernel: int, h: int) -> np.ndarray:
    kernel = ensure_odd(kernel)
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


def process_sequence(sequence_dir: Path, config: PipelineConfig) -> int:
    frames = list_frames(sequence_dir, config.image_exts)
    if not frames:
        return 0

    seq_name = sequence_dir.name
    enhanced_dir = config.output_root / "enhanced" / seq_name
    enhanced_dir.mkdir(parents=True, exist_ok=True)
    nuc = SceneBasedNUC(alpha=config.nuc_alpha)
    for frame_path in frames:
        gray = read_gray_image(frame_path)
        corrected = nuc.apply(gray)
        denoised = denoise_frame(corrected, config.denoise_method, config.denoise_kernel, config.denoise_h)
        enhanced = apply_clahe_and_gamma(
            denoised,
            clip_limit=config.clahe_clip_limit,
            tile_grid_size=config.clahe_tile_grid_size,
            gamma=config.gamma,
        )
        enhanced_path = enhanced_dir / frame_path.name
        cv2.imwrite(str(enhanced_path), enhanced)

    print(f"[sequence] {sequence_dir.name}: frames={len(frames)}")
    return len(frames)


def run_pipeline(config: PipelineConfig) -> None:
    input_root = Path(config.input_root)
    if not input_root.exists():
        raise FileNotFoundError(f"No sequence folders found under: {input_root}")

    config.output_root.mkdir(parents=True, exist_ok=True)
    total_frames = 0
    direct_count = process_sequence(input_root, config)
    if direct_count > 0:
        total_frames += direct_count
    else:
        for sequence_dir in input_root.iterdir():
            if sequence_dir.is_dir():
                frame_count = process_sequence(sequence_dir, config)
                total_frames += frame_count

    print(f"Total frames processed: {total_frames}")
    print(f"Output root: {config.output_root}")


def parse_args() -> Namespace:
    parser = ArgumentParser(description="Infrared preprocess pipeline")
    parser.add_argument("--config", type=str, default="dataset/preprocess_config.yaml", help="配置文件路径")
    parser.add_argument("--input", type=str, default=None, help="覆盖 input_root")
    parser.add_argument("--output", type=str, default=None, help="覆盖 output_root")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(Path(args.config))

    if args.input:
        cfg.input_root = Path(args.input).resolve()
    if args.output:
        cfg.output_root = Path(args.output).resolve()

    run_pipeline(cfg)


if __name__ == "__main__":
    main()

