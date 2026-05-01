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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="独立评估接口：YOLO mAP + spotGEO + FPS")
    
    parser.add_argument("--weights", type=str, default=None,
                        help="模型权重路径，如 runs/detect/exp_xxx/weights/best.pt")
    parser.add_argument("--data", type=str, nargs="+", default=None,
                        help="评估数据目录，支持多个目录")
    parser.add_argument("--config", type=str, default=None,
                        help="训练配置文件（从配置中读取 weights 和 data）")
    parser.add_argument("--preprocess", action="store_true",
                        help="启用预处理（如 NUC+去噪+CLAHE+Gamma）")
    parser.add_argument("--preprocess-config", type=str, default="dataset/dataset_config.yaml",
                        help="预处理配置文件路径")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="推理批量大小（默认 32）")
    parser.add_argument("--img-size", type=int, default=640,
                        help="图像输入尺寸（默认 640）")
    parser.add_argument("--output-dir", type=str, default="runs/evaluate",
                        help="结果输出目录（默认 runs/evaluate）")
    parser.add_argument("--max-visualize", type=int, default=20,
                        help="最大可视化图像数（默认 20）")
    parser.add_argument("--conf", type=float, default=0.1,
                        help="检测置信度阈值（默认 0.1）")
    parser.add_argument("--name", type=str, default=None,
                        help="实验名称（默认自动生成）")
    parser.add_argument("--true-labels", type=str, default="dataset/true_labels.json",
                        help="真实标签 JSON 路径（默认 dataset/true_labels.json）")
    
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
    print(f"\n========== FPS 性能指标 ==========")
    print(f"  总图像数      : {fps_metrics['total_images']}")
    print(f"  批量大小      : {fps_metrics['batch_size']}")
    print(f"  总推理时间    : {fps_metrics['total_inference_time_s']:.3f} s")
    print(f"  平均延迟/张   : {fps_metrics['avg_latency_ms']} ms")
    print(f"  平均 FPS      : {fps_metrics['fps']}")
    print(f"===================================\n")


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
    print(f"\n======================================")
    print(f"🌟 spotGEO 竞赛指标 🌟")
    print(f"  Precision : {spotgeo['spotgeo_precision']}")
    print(f"  Recall    : {spotgeo['spotgeo_recall']}")
    print(f"  F1        : {spotgeo['spotgeo_f1']}")
    print(f"  Score(1-F1): {spotgeo['spotgeo_score']}")
    print(f"  MSE       : {spotgeo['spotgeo_mse']}")
    print(f"======================================\n")


def _print_seq_stats(validation_data: list) -> None:
    """打印序列级检测统计"""
    seq_stats = _compute_sequence_stats(validation_data)
    print("\n========== 序列级检测统计 ==========")
    print(f"  {'序列ID':<12} {'总帧数':<10} {'有检测帧':<10} {'检测率':<10} {'总目标数':<10}")
    print("  " + "-" * 52)
    total_frames_with_det = 0
    total_detections = 0
    for seq in seq_stats:
        print(f"  {seq['sequence_id']:<12} {seq['total_frames']:<10} "
              f"{seq['frames_with_detections']:<10} "
              f"{seq['detection_rate']:<10.2%} {seq['total_detections']:<10}")
        total_frames_with_det += seq['frames_with_detections']
        total_detections += seq['total_detections']
    print("  " + "-" * 52)
    print(f"  {'合计':<12} {'':<10} {total_frames_with_det:<10} {'':<10} {total_detections:<10}")
    print("====================================\n")


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
    print(f"[Evaluate] 模型: {weights_path} | 设备: {device} | 预处理: {'开启' if enable_preprocess else '关闭'}")

    # ===== 1. 推理 + 计时 =====
    print("\n[Evaluate] 开始推理...")
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

    # ===== 4. 序列统计打印 =====
    _print_seq_stats(validation_data)

    # ===== 5. 保存结果 =====
    if not no_save:
        run_name = run_name or _resolve_run_name(weights_path)
        save_dir = Path(output_dir) / run_name
        save_dir.mkdir(parents=True, exist_ok=True)

        # JSON
        json_path = save_dir / "spotgeo_predictions.json"
        with json_path.open("w", encoding="utf-8") as f:
            json.dump(validation_data, f, indent=4, ensure_ascii=False)
        print(f"[Evaluate] spotGEO JSON 已保存: {json_path}")

        # 可视化
        viz_dir = save_dir / "visualizations"
        viz_dir.mkdir(parents=True, exist_ok=True)
        n_viz = visualize_detections(predictions, out_dir=viz_dir, max_images=max_visualize, conf=conf)
        print(f"[Evaluate] 可视化: 已保存 {n_viz} 张到 {viz_dir}")

        # 序列统计
        seq_stats = _compute_sequence_stats(validation_data)
        stats_path = save_dir / "sequence_stats.json"
        with stats_path.open("w", encoding="utf-8") as f:
            json.dump(seq_stats, f, indent=2, ensure_ascii=False)
        print(f"[Evaluate] 序列统计已保存: {stats_path}")

        # 综合指标
        metrics = {
            "model": str(weights_path),
            "data": [str(s) for s in source_dirs],
            "enable_preprocess": enable_preprocess,
            **fps_metrics,
            **spotgeo_metrics,
        }
        metrics_path = save_dir / "metrics.json"
        with metrics_path.open("w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        print(f"[Evaluate] 完整指标已保存: {metrics_path}")
    else:
        metrics = {
            "model": str(weights_path),
            "data": [str(s) for s in source_dirs],
            "enable_preprocess": enable_preprocess,
            **fps_metrics,
            **spotgeo_metrics,
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
        print("[Evaluate] 未找到图像文件")
        return [], [], 0

    predictions = []
    inference_times = []
    total_images = 0

    # 预热
    warmup_imgs = []
    for p in all_paths[:20]:
        img = cv2.imread(str(p))
        if img is not None:
            warmup_imgs.append(img)
    if warmup_imgs:
        _ = model(warmup_imgs, verbose=False)

    # 批量推理
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

    # 预热（取第一个 batch）
    for batch in loader:
        warmup_imgs = _extract_images_from_batch(batch)
        if warmup_imgs is not None:
            _ = model(warmup_imgs.to(device), verbose=False)
        break

    # 正式推理
    for batch in loader:
        images = _extract_images_from_batch(batch)
        im_files = _extract_file_names_from_batch(batch)
        if images is None:
            continue

        images = images.to(device)
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
        n_pred = best_result.boxes is not None and len(best_result.boxes.xyxy) or 0
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
                print(f"错误: 在 {output_dir} 下未找到 exp* 目录")
                return

        sources = cfg.get("val_data_dir", [])
        if isinstance(sources, str):
            sources = [sources]
        print(f"[evaluate.py] 从配置文件读取: weights={weights_path}, data={sources}")
    else:
        print("错误: 请提供 --weights + --data 或 --config")
        print("示例: python evaluate.py --weights runs/detect/exp/weights/best.pt --data D:/Dataset/val")
        return

    # 检查权重是否存在
    if not Path(weights_path).exists():
        print(f"错误: 权重文件不存在: {weights_path}")
        print("请通过 --weights 手动指定正确的权重路径")
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
