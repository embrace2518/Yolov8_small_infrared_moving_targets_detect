"""
独立评估接口。

用于在训练结束后（或训练中）随时对任意模型权重进行评估。
支持：
  - 单目录 / 多目录输入
  - 有预处理 / 无预处理对比
  - YOLO mAP + spotGEO 竞赛指标 + FPS
  - 序列级可视化（带置信度、标签统计）
  - 结果一键保存

Usage:
    # 评估单个模型
    python evaluate.py --weights runs/detect/exp_xxx/weights/best.pt --data D:/Dataset/val
    
    # 带预处理对比
    python evaluate.py --weights runs/detect/exp_xxx/weights/best.pt --data D:/Dataset/val --preprocess
    
    # 评估多个目录
    python evaluate.py --weights runs/detect/exp_xxx/weights/best.pt --data D:/Dataset/val/11 D:/Dataset/val/12
    
    # 从训练配置自动加载
    python evaluate.py --config train_config.yaml
"""

from __future__ import annotations

import argparse
import json
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from ultralytics import YOLO

from dataset.dataset import DatasetConfig, YOLODataset, yolo_collate_fn_with_indices
from dataset.preprocess import UnifiedConfig, load_config as load_preprocess_config
from validation import (
    flat_to_hierarchical,
    score_sequences,
    _to_int_or_self,
    _resolve_sources,
)

from utils.logging import get_logger

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="独立评估接口：YOLO mAP + spotGEO + FPS")

    parser.add_argument("--weights", type=str, default=None,
                        help="模型权重路径 (.pt)")
    parser.add_argument("--conf", type=float, default=0.1,
                        help="检测置信度阈值（默认 0.1）")
    parser.add_argument("--preprocess", action="store_true",
                        help="启用预处理（NUC+去噪+CLAHE+Gamma）")
    parser.add_argument("--output-dir", type=str, default="runs/evaluate",
                        help="输出目录")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="推理批量大小（默认 32）")
    parser.add_argument("--img-size", type=int, default=640,
                        help="图像输入尺寸（默认 640）")
    parser.add_argument("--data", type=str, nargs="+", default=None,
                        help="评估数据目录，支持多个目录")
    parser.add_argument("--config", type=str, default=None,
                        help="训练配置文件（从配置中读取 weights 和 data）")
    parser.add_argument("--preprocess-config", type=str, default="dataset/dataset_config.yaml",
                        help="预处理配置文件路径")
    parser.add_argument("--max-visualize", type=int, default=20,
                        help="最大可视化图像数（默认 20）")
    parser.add_argument("--name", type=str, default=None,
                        help="实验名称（默认自动生成）")
    parser.add_argument("--true-labels", type=str, default="dataset/true_labels.json",
                        help="真实标签 JSON 路径（默认 dataset/true_labels.json）")
    parser.add_argument("--track", action="store_true",
                        help="启用轨迹关联跟踪")
    parser.add_argument("--min-hits", type=int, default=3,
                        help="轨迹确认所需最小连续命中帧数（默认 3）")
    parser.add_argument("--max-age", type=int, default=10,
                        help="轨迹丢失后保留的最大帧数（默认 10）")

    return parser.parse_args()


def _resolve_run_name(weights: Path) -> str:
    """从权重路径自动生成实验名称"""
    exp_name = weights.parent.parent.name
    return f"eval_{exp_name}_{time.strftime('%Y%m%d_%H%M%S')}"


def _time_batch(model: YOLO, batch_input, conf: float, device: str) -> tuple:
    """
    执行一个 batch 推理并计时（只计时模型推理，排除数据加载）。

    Returns:
        (results, elapsed_seconds)
    """
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t_start = time.perf_counter()
    batch_results = model(batch_input, verbose=False, conf=conf)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t_end = time.perf_counter()
    return batch_results, t_end - t_start


def _compute_fps_metrics(inference_times: list[float], total_images: int, batch_size: int) -> dict:
    """计算 FPS 和延迟指标"""
    total_time = sum(inference_times)
    avg_latency_ms = (total_time / total_images * 1000) if total_images > 0 else 0.0
    fps = 1000.0 / avg_latency_ms if avg_latency_ms > 0 else 0.0
    return {
        "total_images": total_images,
        "batch_size": batch_size,
        "total_inference_time_s": float(f"{total_time:.3f}"),
        "avg_latency_ms": float(f"{avg_latency_ms:.2f}"),
        "fps": float(f"{fps:.2f}"),
    }


def _print_fps(fps_metrics: dict) -> None:
    logger.info("========== FPS 性能指标 ==========")
    logger.info("  总图像数      : %s", fps_metrics['total_images'])
    logger.info("  批量大小      : %s", fps_metrics['batch_size'])
    logger.info("  总推理时间    : %.3f s", fps_metrics['total_inference_time_s'])
    logger.info("  平均延迟/张   : %s ms", fps_metrics['avg_latency_ms'])
    logger.info("  平均 FPS      : %s", fps_metrics['fps'])
    logger.info("===================================")


def _compute_spotgeo_score(validation_data: list, true_labels_path: str | Path = "dataset/true_labels.json") -> dict:
    """计算 spotGEO 竞赛指标"""
    true_labels_path = Path(true_labels_path)
    if not true_labels_path.exists():
        return {}

    predictions_h = flat_to_hierarchical(validation_data)
    with open(true_labels_path, 'rt') as fp:
        labels_h = flat_to_hierarchical(json.load(fp))
    precision, recall, F1, mse = score_sequences(predictions_h, labels_h)
    score = 1 - F1
    return {
        "spotgeo_precision": float(f"{precision:.4f}"),
        "spotgeo_recall": float(f"{recall:.4f}"),
        "spotgeo_f1": float(f"{F1:.4f}"),
        "spotgeo_score": float(f"{score:.6f}"),
        "spotgeo_mse": float(f"{mse:.4f}"),
    }


def _print_spotgeo(spotgeo: dict) -> None:
    logger.info("======================================")
    logger.info("🌟 spotGEO 竞赛指标 🌟")
    logger.info("  Precision : %s", spotgeo['spotgeo_precision'])
    logger.info("  Recall    : %s", spotgeo['spotgeo_recall'])
    logger.info("  F1        : %s", spotgeo['spotgeo_f1'])
    logger.info("  Score(1-F1): %s", spotgeo['spotgeo_score'])
    logger.info("  MSE       : %s", spotgeo['spotgeo_mse'])
    logger.info("======================================")


def _print_seq_stats(validation_data: list) -> None:
    """打印序列级检测统计"""
    seq_stats = _compute_sequence_stats(validation_data)
    logger.info("========== 序列级检测统计 ==========")
    logger.info("  %-12s %-10s %-10s %-10s %-10s", "序列ID", "总帧数", "有检测帧", "检测率", "总目标数")
    logger.info("  " + "-" * 52)
    total_frames_with_det = 0
    total_detections = 0
    for seq in seq_stats:
        logger.info("  %-12s %-10s %-10s %-10.2f%% %-10s",
                    seq['sequence_id'], seq['total_frames'],
                    seq['frames_with_detections'],
                    seq['detection_rate'] * 100, seq['total_detections'])
        total_frames_with_det += seq['frames_with_detections']
        total_detections += seq['total_detections']
    logger.info("  " + "-" * 52)
    logger.info("  %-12s %-10s %-10s %-10s %-10s", "合计", "", total_frames_with_det, "", total_detections)
    logger.info("====================================")


def _run_inference(
    model: YOLO,
    source_dirs: list[Path],
    enable_preprocess: bool,
    preprocess_config_path: str | Path,
    batch_size: int,
    img_size: int,
    conf: float,
    device: str,
) -> tuple:
    """统一入口：根据是否预处理选择推理路径"""
    if enable_preprocess:
        return _predict_with_dataloader(
            model=model, source_dirs=source_dirs,
            preprocess_config_path=preprocess_config_path,
            img_size=img_size, batch_size=batch_size,
            conf=conf, device=device,
        )
    return _predict_direct(
        model=model, source_dirs=source_dirs,
        batch_size=batch_size, conf=conf, device=device,
    )


def evaluate(
    weights: str | Path,
    sources: str | Path | list[str | Path],
    preprocess_config_path: str | Path = "dataset/dataset_config.yaml",
    enable_preprocess: bool = False,
    batch_size: int = 32,
    img_size: int = 640,
    output_dir: str | Path = "runs/evaluate",
    run_name: Optional[str] = None,
    max_visualize: int = 20,
    conf: float = 0.1,
    no_save: bool = False,
    true_labels_path: str | Path = "dataset/true_labels.json",
    enable_tracking: bool = False,
    min_hits: int = 3,
    max_age: int = 10,
) -> dict:
    """
    核心评估函数：对一个模型在指定数据上做完整评估。

    Args:
        weights: 模型权重路径
        sources: 一个或多个数据目录/文件
        preprocess_config_path: 预处理配置文件
        enable_preprocess: 是否启用预处理
        batch_size: 推理批量大小
        img_size: 图像输入尺寸
        output_dir: 结果输出目录
        run_name: 实验名称
        max_visualize: 最大可视化图像数
        conf: 检测置信度阈值
        no_save: 不保存结果到磁盘
        true_labels_path: 真实标签 JSON 路径（spotGEO 评分用）
        enable_tracking: 是否启用轨迹关联跟踪
        min_hits: 轨迹确认所需最小连续命中帧数
        max_age: 轨迹丢失后保留的最大帧数

    Returns:
        metrics: 包含所有评估指标的字典
    """
    # ===== 初始化 =====
    weights_path = Path(weights)
    if isinstance(sources, (str, Path)):
        sources = [sources]
    source_dirs = [Path(s) for s in sources]

    model = YOLO(str(weights_path))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("模型: %s | 设备: %s | 预处理: %s", weights_path, device, "开启" if enable_preprocess else "关闭")

    # ===== 1. 推理 + 计时 =====
    logger.info("开始推理...")
    predictions, inference_times, total_images = _run_inference(
        model=model, source_dirs=source_dirs,
        enable_preprocess=enable_preprocess,
        preprocess_config_path=preprocess_config_path,
        batch_size=batch_size, img_size=img_size,
        conf=conf, device=device,
    )

    # ===== 2. FPS 指标 =====
    fps_metrics = _compute_fps_metrics(inference_times, total_images, batch_size)
    _print_fps(fps_metrics)

    # ===== 3. spotGEO 评分 =====
    validation_data = _predictions_to_json(predictions)
    spotgeo_metrics = _compute_spotgeo_score(validation_data, true_labels_path=true_labels_path)
    if spotgeo_metrics:
        _print_spotgeo(spotgeo_metrics)

    # ===== 4. 轨迹关联跟踪（可选）=====
    tracking_data = []
    tracking_metrics = {}
    if enable_tracking:
        logger.info("")
        logger.info("=" * 70)
        logger.info("开始轨迹关联跟踪...")
        logger.info("=" * 70)
        tracking_data, tracking_metrics = _run_tracking(
            predictions, min_hits=min_hits, max_age=max_age,
        )
        _print_tracking_metrics(tracking_metrics)

    # ===== 5. 序列统计打印 =====
    _print_seq_stats(validation_data)

    # ===== 6. 保存结果 =====
    if not no_save:
        run_name = run_name or _resolve_run_name(weights_path)
        save_dir = Path(output_dir) / run_name
        save_dir.mkdir(parents=True, exist_ok=True)

        # JSON
        json_path = save_dir / "spotgeo_predictions.json"
        with json_path.open("w", encoding="utf-8") as f:
            json.dump(validation_data, f, indent=4, ensure_ascii=False)
        logger.info("spotGEO JSON 已保存: %s", json_path)

        # 可视化
        viz_dir = save_dir / "visualizations"
        viz_dir.mkdir(parents=True, exist_ok=True)
        n_viz = visualize_detections(predictions, out_dir=viz_dir, max_images=max_visualize, conf=conf)
        logger.info("可视化: 已保存 %s 张到 %s", n_viz, viz_dir)

        # 序列统计
        seq_stats = _compute_sequence_stats(validation_data)
        stats_path = save_dir / "sequence_stats.json"
        with stats_path.open("w", encoding="utf-8") as f:
            json.dump(seq_stats, f, indent=2, ensure_ascii=False)
        logger.info("序列统计已保存: %s", stats_path)

        # 综合指标
        metrics = {
            "model": str(weights_path),
            "data": [str(s) for s in source_dirs],
            "enable_preprocess": enable_preprocess,
            "enable_tracking": enable_tracking,
            **fps_metrics,
            **spotgeo_metrics,
            **tracking_metrics,
        }
        metrics_path = save_dir / "metrics.json"
        with metrics_path.open("w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        logger.info("完整指标已保存: %s", metrics_path)

        # 跟踪结果 JSON
        if tracking_data:
            track_path = save_dir / "tracking_results.json"
            with track_path.open("w", encoding="utf-8") as f:
                json.dump(tracking_data, f, indent=2, ensure_ascii=False)
            logger.info("跟踪结果已保存: %s", track_path)
    else:
        metrics = {
            "model": str(weights_path),
            "data": [str(s) for s in source_dirs],
            "enable_preprocess": enable_preprocess,
            "enable_tracking": enable_tracking,
            **fps_metrics,
            **spotgeo_metrics,
            **tracking_metrics,
        }

    return metrics


# ===================== 内部辅助函数 =====================


def _predict_direct(
    model: YOLO,
    source_dirs: list[Path],
    batch_size: int = 32,
    conf: float = 0.1,
    device: str = "cuda",
) -> tuple:
    """不使用预处理，直接目录推理（批量 + CUDA 同步计时）"""
    all_paths = _resolve_sources(source_dirs)
    if not all_paths:
        logger.warning("未找到图像文件")
        return [], [], 0

    predictions = []
    inference_times = []
    total_images = 0
    warmup_done = False

    # 批量推理（首个 batch 同时用于预热）
    for start in range(0, len(all_paths), batch_size):
        batch_paths = all_paths[start:start + batch_size]
        batch_images = []
        batch_orig_paths = []
        for p in batch_paths:
            img = cv2.imread(str(p))
            if img is not None:
                batch_images.append(img)
                batch_orig_paths.append(p)

        if not batch_images:
            continue

        # 使用第一个有效 batch 做预热，不重复加载
        if not warmup_done:
            _ = model(batch_images, verbose=False)
            warmup_done = True

        total_images += len(batch_images)
        batch_results, elapsed = _time_batch(model, batch_images, conf=conf, device=device)
        inference_times.append(elapsed)

        for idx, result in enumerate(batch_results):
            if idx < len(batch_orig_paths):
                result.path = str(batch_orig_paths[idx])
            predictions.append(result)

    return predictions, inference_times, total_images


def _predict_with_dataloader(
    model: YOLO,
    source_dirs: list[Path],
    preprocess_config_path: str | Path,
    img_size: int = 640,
    batch_size: int = 32,
    conf: float = 0.1,
    device: str = "cuda",
) -> tuple:
    """使用 DataLoader（带预处理）推理 + 计时"""
    unified_config: UnifiedConfig = load_preprocess_config(preprocess_config_path)
    unified_config.enable_preprocess = True
    unified_config.enable_augmentation = False

    dataset = YOLODataset(
        DatasetConfig(
            images_dir=source_dirs,
            labels_dir=source_dirs,
            unified_config=unified_config,
            target_size=(img_size, img_size),
        ),
        mode="val",
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.startswith("cuda"),
        collate_fn=yolo_collate_fn_with_indices,
    )

    predictions = []
    inference_times = []
    total_images = 0

    # 正式推理（首个 batch 同时用于预热）
    warmup_done = False
    for batch in loader:
        images = _extract_images_from_batch(batch)
        im_files = _extract_file_names_from_batch(batch)
        if images is None:
            continue

        images = images.to(device)

        # 使用第一个有效 batch 做预热，避免重复加载
        if not warmup_done:
            _ = model(images, verbose=False)
            warmup_done = True

        total_images += images.size(0)
        batch_results, elapsed = _time_batch(model, images, conf=conf, device=device)
        inference_times.append(elapsed)

        for idx, result in enumerate(batch_results):
            if idx < len(im_files):
                result.path = im_files[idx]
            predictions.append(result)

    return predictions, inference_times, total_images


def _extract_images_from_batch(batch) -> torch.Tensor | None:
    """从 DataLoader batch 中提取图像 tensor"""
    if isinstance(batch, dict):
        return batch.get("img", None)
    elif isinstance(batch, (list, tuple)):
        return batch[0] if batch else None
    return None


def _extract_file_names_from_batch(batch) -> list[str]:
    """从 DataLoader batch 中提取文件路径"""
    if isinstance(batch, dict):
        return batch.get("im_file", [])
    elif isinstance(batch, (list, tuple)) and len(batch) >= 3:
        return [f"sample_{i}" for i in range(batch[0].size(0))]
    return []


def _predictions_to_json(predictions: list) -> list:
    """将 YOLO 预测结果转换为 spotGEO JSON 格式"""
    validation_data = []
    for result in predictions:
        img_path = Path(str(result.path))
        parent_name = img_path.parent.name
        stem_name = img_path.stem

        seq_match = re.search(r"(\d+)", parent_name)
        frame_match = re.search(r"(\d+)", stem_name)
        sequence_id = _to_int_or_self(seq_match.group(1)) if seq_match else parent_name
        frame = _to_int_or_self(frame_match.group(1)) if frame_match else stem_name

        object_coords = []
        boxes = result.boxes
        if boxes is not None:
            for box in boxes.xyxy:
                x1, y1, x2, y2 = box.tolist()
                x_center = (x1 + x2) / 2.0
                y_center = (y1 + y2) / 2.0
                object_coords.append([x_center, y_center])

        validation_data.append({
            "sequence_id": sequence_id,
            "frame": frame,
            "num_objects": len(object_coords),
            "object_coords": object_coords,
        })
    return validation_data


def _run_tracking(
    predictions: list,
    min_hits: int = 3,
    max_age: int = 10,
) -> tuple[list, dict]:
    """
    按序列运行 SORT 跟踪器。

    Returns:
        tracking_data: 扁平化的跟踪结果列表（同 spotGEO 格式但含 track_id）
        tracking_metrics: 汇总指标
    """
    from mot import SORTTracker, compute_mot_metrics

    # 按序列分组
    seq_results: dict[str, list] = {}
    for result in predictions:
        img_path = Path(str(result.path))
        parent_name = img_path.parent.name
        seq_match = re.search(r"(\d+)", parent_name)
        seq_id = seq_match.group(1) if seq_match else parent_name
        seq_results.setdefault(seq_id, []).append(result)

    all_track_data: list[dict] = []
    seq_track_results: list[list[dict]] = []
    seq_gt_labels: list[list[dict]] = []
    total_trajectories = 0

    tracker = SORTTracker(min_hits=min_hits, max_age=max_age)

    for seq_id in sorted(seq_results.keys(), key=lambda x: int(x) if x.isdigit() else x):
        seq_frames = seq_results[seq_id]
        tracker.reset()
        seq_tracks: list[list[dict]] = []
        seq_gt: list[list[dict]] = []

        for frame_idx, result in enumerate(seq_frames):
            # Extract detections [N, 4] in xyxy
            boxes = result.boxes
            if boxes is not None and len(boxes.xyxy) > 0:
                dets = boxes.xyxy.cpu().numpy()
            else:
                dets = np.zeros((0, 4), dtype=np.float32)

            # Run tracker
            tracks = tracker.update(dets, frame_idx)
            seq_tracks.append(tracks)

            # Build output data for this frame
            img_path = Path(str(result.path))
            stem_name = img_path.stem
            frame_match = re.search(r"(\d+)", stem_name)
            frame_num = _to_int_or_self(frame_match.group(1)) if frame_match else frame_idx

            track_coords = []
            for t in tracks:
                if t["confirmed"]:
                    bbox = t["bbox"]
                    track_coords.append({
                        "track_id": t["id"],
                        "cx": float((bbox[0] + bbox[2]) / 2.0),
                        "cy": float((bbox[1] + bbox[3]) / 2.0),
                    })

            all_track_data.append({
                "sequence_id": seq_id,
                "frame": frame_num,
                "num_tracks": len(track_coords),
                "tracks": track_coords,
            })

            # Ground truth for this frame
            gt_boxes = []
            label_path = img_path.with_suffix(".txt")
            if label_path.exists():
                try:
                    with open(label_path, "r") as f:
                        for line in f:
                            parts = line.strip().split()
                            if len(parts) >= 5:
                                _, xc, yc, bw, bh = map(float, parts[:5])
                                gt_boxes.append({
                                    "bbox": [
                                        xc - bw / 2, yc - bh / 2,
                                        xc + bw / 2, yc + bh / 2,
                                    ],
                                })
                except Exception:
                    pass
            seq_gt.append(gt_boxes)

        seq_track_results.append(seq_tracks)
        seq_gt_labels.append(seq_gt)
        total_trajectories += tracker.n_confirmed

    # Compute MOT metrics across all sequences
    mot_metrics: dict = {}
    if seq_track_results and seq_gt_labels:
        try:
            mot_metrics = compute_mot_metrics(seq_track_results, seq_gt_labels)
        except Exception as exc:
            logger.warning("MOT metrics calculation failed: %s", exc)

    mot_metrics["n_trajectories"] = total_trajectories
    return all_track_data, mot_metrics


def _print_tracking_metrics(metrics: dict) -> None:
    logger.info("========== 轨迹关联指标 ==========")
    logger.info("  MOTA         : %s", metrics.get('mota', 'N/A'))
    logger.info("  MOTP         : %s", metrics.get('motp', 'N/A'))
    logger.info("  IDF1         : %s", metrics.get('idf1', 'N/A'))
    logger.info("  ID Switches  : %s", metrics.get('id_switches', 'N/A'))
    logger.info("  TP/FP/FN     : %s / %s / %s",
                metrics.get('total_tp', 'N/A'),
                metrics.get('total_fp', 'N/A'),
                metrics.get('total_fn', 'N/A'))
    logger.info("  确认轨迹数    : %s", metrics.get('n_trajectories', 'N/A'))
    logger.info("====================================")


def _draw_trajectory_overlay(
    img: np.ndarray,
    track_history: dict[int, list[tuple[float, float]]],
    active_ids: set[int],
    colors: dict[int, tuple[int, int, int]],
) -> np.ndarray:
    """在图像上绘制轨迹路径和当前框。"""
    for tid, positions in track_history.items():
        if len(positions) < 2:
            continue
        color = colors.get(tid, (255, 255, 0))
        # Draw path
        for i in range(1, len(positions)):
            pt1 = (int(positions[i - 1][0]), int(positions[i - 1][1]))
            pt2 = (int(positions[i][0]), int(positions[i][1]))
            fade = 0.3 + 0.7 * (i / len(positions))
            faded_color = tuple(int(c * fade) for c in color)
            cv2.line(img, pt1, pt2, faded_color, 1, cv2.LINE_AA)
        # Mark current position
        if tid in active_ids:
            cx, cy = int(positions[-1][0]), int(positions[-1][1])
            cv2.circle(img, (cx, cy), 4, color, -1)
            cv2.putText(img, f"ID:{tid}", (cx + 5, cy - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    return img


def visualize_detections(
    predictions: list,
    out_dir: Path | str,
    max_images: int = 20,
    conf: float = 0.1,
) -> int:
    """
    增强的可视化：每个序列保存 1 张代表性图像，
    包含置信度标注、真实框/预测框对比、统计信息叠加。

    Returns:
        保存的图像数量
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 按序列分组
    seq_groups = {}
    for result in predictions:
        img_path = Path(str(result.path))
        parent_name = img_path.parent.name
        seq_match = re.search(r"(\d+)", parent_name)
        sequence_id = seq_match.group(1) if seq_match else parent_name
        seq_groups.setdefault(sequence_id, []).append(result)

    saved_count = 0
    for seq_id, seq_results in seq_groups.items():
        if saved_count >= max_images:
            break

        # 找有检测且置信度最高的帧
        best_result = None
        best_conf = 0
        for result in seq_results:
            boxes = result.boxes
            if boxes is not None and len(boxes) > 0:
                confs = boxes.conf.cpu().numpy() if hasattr(boxes.conf, 'cpu') else np.array(boxes.conf)
                max_c = float(confs.max()) if len(confs) > 0 else 0
                if max_c > best_conf:
                    best_conf = max_c
                    best_result = result

        # 如果全无检测，选第一帧
        if best_result is None and len(seq_results) > 0:
            best_result = seq_results[0]

        if best_result is None:
            continue

        img_path = Path(str(best_result.path))
        stem_name = img_path.stem

        # 复制图像
        img = best_result.orig_img.copy()
        h, w = img.shape[:2]

        # ---- 统计信息 ----
        n_truth = 0
        n_pred = len(best_result.boxes.xyxy) if best_result.boxes is not None else 0
        pred_confs = []
        tp_count = 0

        # 绘制真实标签框（绿色）
        label_path = img_path.with_suffix('.txt')
        if label_path.exists():
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        _, xc, yc, bw, bh = map(float, parts[:5])
                        tx1 = int((xc - bw / 2) * w)
                        ty1 = int((yc - bh / 2) * h)
                        tx2 = int((xc + bw / 2) * w)
                        ty2 = int((yc + bh / 2) * h)
                        cv2.rectangle(img, (tx1, ty1), (tx2, ty2), (0, 255, 0), 2)
                        cv2.putText(img, "GT", (tx1, max(ty1 - 5, 10)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                        n_truth += 1

        # 绘制预测框（红色，带置信度）
        if best_result.boxes is not None:
            for i, box in enumerate(best_result.boxes.xyxy):
                x1, y1, x2, y2 = map(int, box.tolist())
                conf_val = float(best_result.boxes.conf[i]) if best_result.boxes.conf is not None else 1.0
                pred_confs.append(conf_val)

                # 颜色：高置信度用亮红，低置信度用暗红
                color = (0, 0, 255) if conf_val > 0.5 else (0, 100, 200)
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                cv2.putText(img, f"Pred {conf_val:.2f}", (x1, max(y1 - 25, 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        # ---- 信息叠加 ----
        info_lines = [
            f"Seq: {seq_id} | Frame: {stem_name}",
            f"GT: {n_truth} | Pred: {n_pred}",
        ]
        if pred_confs:
            info_lines.append(f"Conf: mean={np.mean(pred_confs):.2f} max={max(pred_confs):.2f}")

        for i, line in enumerate(info_lines):
            cv2.putText(img, line, (10, 30 + i * 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(img, line, (10, 30 + i * 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

        status = "detected" if n_pred > 0 else "no_det"
        out_file = out_dir / f"seq{seq_id}_{stem_name}_{status}.jpg"
        cv2.imwrite(str(out_file), img)
        saved_count += 1

    return saved_count


def _compute_sequence_stats(validation_data: list) -> list:
    """计算每个序列的检测统计信息"""
    seq_frames: dict = defaultdict(list)
    for item in validation_data:
        seq_id = str(item["sequence_id"])
        seq_frames[seq_id].append(item)

    stats = []
    for seq_id in sorted(seq_frames.keys(), key=lambda x: int(x) if x.isdigit() else x):
        frames = seq_frames[seq_id]
        total_frames = len(frames)
        frames_with_det = sum(1 for f in frames if f["num_objects"] > 0)
        total_detections = sum(f["num_objects"] for f in frames)

        stats.append({
            "sequence_id": seq_id,
            "total_frames": total_frames,
            "frames_with_detections": frames_with_det,
            "detection_rate": float(f"{frames_with_det / total_frames:.4f}") if total_frames > 0 else 0.0,
            "total_detections": total_detections,
        })

    return stats


def main():
    args = parse_args()

    # 优先级: --weights > --config
    if args.weights and args.data:
        weights_path = args.weights
        sources = args.data
    elif args.config:
        # 从训练配置中读取
        with open(args.config, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        output_dir = Path(cfg.get("output_dir", "runs/detect"))
        run_name = cfg.get("run_name", None)

        if run_name:
            # 有 run_name：runs/detect/exp_xxx/weights/best.pt
            weights_path = str(output_dir / run_name / "weights" / "best.pt")
        else:
            # 没有 run_name：取 output_dir 下最新 exp 目录
            exp_dirs = sorted(output_dir.glob("exp*"))
            if exp_dirs:
                weights_path = str(exp_dirs[-1] / "weights" / "best.pt")
            else:
                logger.error("在 %s 下未找到 exp* 目录", output_dir)
                return

        sources = cfg.get("val_data_dir", [])
        if isinstance(sources, str):
            sources = [sources]
        logger.info("从配置文件读取: weights=%s, data=%s", weights_path, sources)
    else:
        logger.error("请提供 --weights + --data 或 --config")
        logger.error("示例: python evaluate.py --weights runs/detect/exp/weights/best.pt --data D:/Dataset/val")
        return

    # 检查权重是否存在
    if not Path(weights_path).exists():
        logger.error("权重文件不存在: %s", weights_path)
        logger.error("请通过 --weights 手动指定正确的权重路径")
        return

    metrics = evaluate(
        weights=weights_path,
        sources=sources,
        preprocess_config_path=args.preprocess_config,
        enable_preprocess=args.preprocess,
        batch_size=args.batch_size,
        img_size=args.img_size,
        output_dir=args.output_dir,
        run_name=args.name,
        max_visualize=args.max_visualize,
        conf=args.conf,
        true_labels_path=args.true_labels,
        enable_tracking=args.track,
        min_hits=args.min_hits,
        max_age=args.max_age,
    )

    print("\n" + "=" * 60)
    print("评估完成！关键指标:")
    print(f"  FPS          : {metrics.get('fps', 'N/A')}")
    print(f"  延迟          : {metrics.get('avg_latency_ms', 'N/A')} ms")
    print(f"  spotGEO Score: {metrics.get('spotgeo_score', 'N/A')}")
    print(f"  spotGEO MSE  : {metrics.get('spotgeo_mse', 'N/A')}")
    print("=" * 60)


if __name__ == "__main__":
    main()
