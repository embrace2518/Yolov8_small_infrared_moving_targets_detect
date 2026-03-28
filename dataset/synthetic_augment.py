import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


@dataclass
class SyntheticConfig:
    enhanced_root: Path
    output_root: Path
    image_exts: tuple[str, ...]
    gmm_history: int
    gmm_var_threshold: float
    gmm_detect_shadows: bool
    learning_rate: float
    min_area: int
    max_area: int
    max_proposals_per_frame: int
    binary_threshold: int
    morph_kernel: int
    save_masks: bool
    save_overlays: bool
    save_metadata: bool
    enable_synthetic: bool
    copies_per_frame: int
    blend_alpha: float
    max_shift: int


def natural_key(path: Path) -> tuple[int, Any]:
    stem = path.stem
    return (0, int(stem)) if stem.isdigit() else (1, stem.lower())


def ensure_odd(kernel: int) -> int:
    kernel = max(1, int(kernel))
    return kernel if kernel % 2 == 1 else kernel + 1


def load_config(config_path: Path) -> SyntheticConfig:
    with config_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    io_cfg = raw.get("io", {})
    gmm_cfg = raw.get("gmm", {})
    proposal_cfg = raw.get("proposal", {})
    output_cfg = raw.get("output", {})
    synthetic_cfg = raw.get("synthetic", {})

    output_root = Path(io_cfg.get("output_root", "D:/Dataset_preprocessed/train")).resolve()
    enhanced_root = Path(io_cfg.get("enhanced_root", str(output_root / "enhanced"))).resolve()

    return SyntheticConfig(
        enhanced_root=enhanced_root,
        output_root=output_root,
        image_exts=tuple(e.lower() for e in io_cfg.get("image_exts", list(IMAGE_EXTS))),
        gmm_history=int(gmm_cfg.get("history", 120)),
        gmm_var_threshold=float(gmm_cfg.get("var_threshold", 16.0)),
        gmm_detect_shadows=bool(gmm_cfg.get("detect_shadows", False)),
        learning_rate=float(gmm_cfg.get("learning_rate", -1.0)),
        min_area=int(proposal_cfg.get("min_area", 4)),
        max_area=int(proposal_cfg.get("max_area", 1600)),
        max_proposals_per_frame=int(proposal_cfg.get("max_proposals_per_frame", 10)),
        binary_threshold=int(proposal_cfg.get("binary_threshold", 200)),
        morph_kernel=int(proposal_cfg.get("morph_kernel", 3)),
        save_masks=bool(output_cfg.get("save_masks", True)),
        save_overlays=bool(output_cfg.get("save_overlays", True)),
        save_metadata=bool(output_cfg.get("save_metadata", True)),
        enable_synthetic=bool(synthetic_cfg.get("enable", False)),
        copies_per_frame=max(1, int(synthetic_cfg.get("copies_per_frame", 1))),
        blend_alpha=float(synthetic_cfg.get("blend_alpha", 0.7)),
        max_shift=max(1, int(synthetic_cfg.get("max_shift", 12))),
    )


def list_sequence_dirs(root: Path) -> list[Path]:
    return sorted([p for p in root.iterdir() if p.is_dir()], key=lambda p: p.name.lower())


def list_frames(sequence_dir: Path, image_exts: tuple[str, ...]) -> list[Path]:
    frames = [p for p in sequence_dir.iterdir() if p.is_file() and p.suffix.lower() in image_exts]
    return sorted(frames, key=natural_key)


def read_gray_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Failed to read image: {path}")
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if image.dtype != np.uint8:
        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return image


def suppress_background(
    gmm: cv2.BackgroundSubtractor,
    enhanced: np.ndarray,
    learning_rate: float,
    binary_threshold: int,
    morph_kernel: int,
) -> np.ndarray:
    fg_mask = gmm.apply(enhanced, learningRate=learning_rate)
    _, fg_mask = cv2.threshold(fg_mask, binary_threshold, 255, cv2.THRESH_BINARY)
    kernel_size = ensure_odd(morph_kernel)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
    return fg_mask


def extract_proposals(mask: np.ndarray, enhanced: np.ndarray, min_area: int, max_area: int, max_proposals_per_frame: int) -> list[dict[str, Any]]:
    num_labels, _, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    proposals: list[dict[str, Any]] = []

    for label_idx in range(1, num_labels):
        x, y, w, h, area = stats[label_idx]
        if area < min_area or area > max_area:
            continue

        region = enhanced[y : y + h, x : x + w]
        if region.size == 0:
            continue

        mean_intensity = float(np.mean(region))
        peak_intensity = float(np.max(region))
        score = peak_intensity + 0.2 * mean_intensity + 0.1 * area
        center_x, center_y = centroids[label_idx]
        proposals.append(
            {
                "bbox": [int(x), int(y), int(w), int(h)],
                "area": int(area),
                "center_x": int(round(center_x)),
                "center_y": int(round(center_y)),
                "mean_intensity": round(mean_intensity, 3),
                "peak_intensity": round(peak_intensity, 3),
                "score": round(score, 3),
            }
        )

    proposals.sort(key=lambda item: item["score"], reverse=True)
    return proposals[:max_proposals_per_frame]


def build_overlay(gray: np.ndarray, proposals: list[dict[str, Any]]) -> np.ndarray:
    overlay = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    for proposal in proposals:
        x, y, w, h = proposal["bbox"]
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 255, 0), 1)
        cv2.circle(overlay, (proposal["center_x"], proposal["center_y"]), 2, (0, 0, 255), -1)
        cv2.putText(
            overlay,
            f"{proposal['score']:.2f}",
            (x, max(y - 4, 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (255, 255, 0),
            1,
            cv2.LINE_AA,
        )
    return overlay


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def labels_find_image(image_dir: Path, stem: str) -> Path | None:
    for ext in IMAGE_EXTS:
        candidate = image_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def labels_clip_box(
    x: int,
    y: int,
    w: int,
    h: int,
    img_w: int,
    img_h: int,
) -> tuple[float, float, float, float] | None:
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(img_w, x + w)
    y2 = min(img_h, y + h)
    if x2 <= x1 or y2 <= y1:
        return None

    bw = float(x2 - x1)
    bh = float(y2 - y1)
    cx = float(x1) + bw / 2.0
    cy = float(y1) + bh / 2.0
    return cx / img_w, cy / img_h, bw / img_w, bh / img_h


def labels_convert_sequence(
    proposal_seq_dir: Path,
    image_seq_dir: Path,
    label_seq_dir: Path,
    class_id: int,
    score_thresh: float,
    max_objects: int,
) -> tuple[int, int]:
    json_files = sorted(proposal_seq_dir.glob("*.json"), key=lambda p: p.stem)
    label_seq_dir.mkdir(parents=True, exist_ok=True)

    converted = 0
    written_boxes = 0
    for proposal_file in json_files:
        with proposal_file.open("r", encoding="utf-8") as f:
            payload: dict[str, Any] = json.load(f)

        frame_stem = Path(payload.get("frame", proposal_file.stem)).stem
        image_path = labels_find_image(image_seq_dir, frame_stem)
        if image_path is None:
            continue

        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            continue
        img_h, img_w = image.shape[:2]

        proposals = payload.get("proposals", [])
        if not isinstance(proposals, list):
            proposals = []

        valid_lines: list[str] = []
        for item in proposals:
            score = float(item.get("score", 0.0))
            if score < score_thresh:
                continue

            bbox = item.get("bbox", [])
            if not isinstance(bbox, list) or len(bbox) != 4:
                continue

            x, y, w, h = [int(v) for v in bbox]
            norm = labels_clip_box(x, y, w, h, img_w, img_h)
            if norm is None:
                continue

            x_n, y_n, w_n, h_n = norm
            valid_lines.append(f"{class_id} {x_n:.6f} {y_n:.6f} {w_n:.6f} {h_n:.6f}")
            if max_objects > 0 and len(valid_lines) >= max_objects:
                break

        label_path = label_seq_dir / f"{frame_stem}.txt"
        with label_path.open("w", encoding="utf-8") as f:
            if valid_lines:
                f.write("\n".join(valid_lines) + "\n")

        converted += 1
        written_boxes += len(valid_lines)

    return converted, written_boxes


def labels_run(
    proposal_root: Path,
    image_root: Path,
    label_root: Path,
    class_id: int,
    score_thresh: float,
    max_objects: int,
) -> None:
    if not proposal_root.exists():
        raise FileNotFoundError(f"Proposal root not found: {proposal_root}")
    if not image_root.exists():
        raise FileNotFoundError(f"Image root not found: {image_root}")

    seq_dirs = sorted([p for p in proposal_root.iterdir() if p.is_dir()], key=lambda p: p.name.lower())
    if not seq_dirs:
        raise FileNotFoundError(f"No sequence directories under: {proposal_root}")

    total_frames = 0
    total_boxes = 0
    for seq_dir in seq_dirs:
        image_seq_dir = image_root / seq_dir.name
        if not image_seq_dir.exists():
            print(f"[skip] missing image dir for sequence: {seq_dir.name}")
            continue

        label_seq_dir = label_root / seq_dir.name
        frames, boxes = labels_convert_sequence(
            proposal_seq_dir=seq_dir,
            image_seq_dir=image_seq_dir,
            label_seq_dir=label_seq_dir,
            class_id=class_id,
            score_thresh=score_thresh,
            max_objects=max_objects,
        )
        total_frames += frames
        total_boxes += boxes
        print(f"[labels] {seq_dir.name}: frames={frames}, boxes={boxes}")

    print(f"Labels done. frames={total_frames}, boxes={total_boxes}")
    print(f"Labels saved to: {label_root}")


def synthesize_from_proposals(
    enhanced: np.ndarray,
    proposals: list[dict[str, Any]],
    rng: np.random.Generator,
    copies_per_frame: int,
    blend_alpha: float,
    max_shift: int,
) -> list[np.ndarray]:
    if not proposals:
        return []

    h, w = enhanced.shape[:2]
    synth_list: list[np.ndarray] = []
    blend_alpha = float(np.clip(blend_alpha, 0.1, 1.0))

    for _ in range(copies_per_frame):
        out = enhanced.copy().astype(np.float32)
        for proposal in proposals[:3]:
            x, y, bw, bh = proposal["bbox"]
            patch = enhanced[y : y + bh, x : x + bw]
            if patch.size == 0:
                continue

            shift_x = int(rng.integers(-max_shift, max_shift + 1))
            shift_y = int(rng.integers(-max_shift, max_shift + 1))
            nx = int(np.clip(x + shift_x, 0, max(0, w - bw)))
            ny = int(np.clip(y + shift_y, 0, max(0, h - bh)))

            target = out[ny : ny + bh, nx : nx + bw]
            if target.shape != patch.shape:
                continue
            out[ny : ny + bh, nx : nx + bw] = (1.0 - blend_alpha) * target + blend_alpha * patch

        synth_list.append(np.clip(out, 0, 255).astype(np.uint8))

    return synth_list


def process_sequence(sequence_dir: Path, config: SyntheticConfig, rng: np.random.Generator) -> tuple[int, int, int]:
    frames = list_frames(sequence_dir, config.image_exts)
    if not frames:
        return 0, 0, 0

    seq_name = sequence_dir.name
    mask_dir = config.output_root / "masks" / seq_name
    overlay_dir = config.output_root / "overlays" / seq_name
    meta_dir = config.output_root / "proposals" / seq_name
    synth_dir = config.output_root / "synthetic" / seq_name

    if config.save_masks:
        mask_dir.mkdir(parents=True, exist_ok=True)
    if config.save_overlays:
        overlay_dir.mkdir(parents=True, exist_ok=True)
    if config.save_metadata:
        meta_dir.mkdir(parents=True, exist_ok=True)
    if config.enable_synthetic:
        synth_dir.mkdir(parents=True, exist_ok=True)

    gmm = cv2.createBackgroundSubtractorMOG2(
        history=config.gmm_history,
        varThreshold=config.gmm_var_threshold,
        detectShadows=config.gmm_detect_shadows,
    )

    total_proposals = 0
    total_synth = 0
    for frame_path in frames:
        enhanced = read_gray_image(frame_path)
        mask = suppress_background(
            gmm,
            enhanced,
            learning_rate=config.learning_rate,
            binary_threshold=config.binary_threshold,
            morph_kernel=config.morph_kernel,
        )
        proposals = extract_proposals(
            mask,
            enhanced,
            min_area=config.min_area,
            max_area=config.max_area,
            max_proposals_per_frame=config.max_proposals_per_frame,
        )
        total_proposals += len(proposals)

        if config.save_masks:
            cv2.imwrite(str(mask_dir / frame_path.name), mask)
        if config.save_overlays:
            overlay = build_overlay(enhanced, proposals)
            cv2.imwrite(str(overlay_dir / frame_path.name), overlay)
        if config.save_metadata:
            payload = {
                "sequence_id": seq_name,
                "frame": frame_path.stem,
                "num_proposals": len(proposals),
                "proposals": proposals,
            }
            save_json(meta_dir / f"{frame_path.stem}.json", payload)

        if config.enable_synthetic:
            synth_list = synthesize_from_proposals(
                enhanced,
                proposals,
                rng=rng,
                copies_per_frame=config.copies_per_frame,
                blend_alpha=config.blend_alpha,
                max_shift=config.max_shift,
            )
            for idx, synth in enumerate(synth_list):
                out_name = f"{frame_path.stem}_syn{idx:02d}{frame_path.suffix}"
                cv2.imwrite(str(synth_dir / out_name), synth)
            total_synth += len(synth_list)

    return len(frames), total_proposals, total_synth


def run_pipeline(config: SyntheticConfig) -> None:
    sequence_dirs = list_sequence_dirs(config.enhanced_root)
    if not sequence_dirs:
        raise FileNotFoundError(f"No sequence folders found under: {config.enhanced_root}")

    rng = np.random.default_rng(2026)
    total_frames = 0
    total_proposals = 0
    total_synth = 0

    for sequence_dir in sequence_dirs:
        frame_count, proposal_count, synth_count = process_sequence(sequence_dir, config, rng)
        total_frames += frame_count
        total_proposals += proposal_count
        total_synth += synth_count
        print(
            f"[sequence] {sequence_dir.name}: frames={frame_count}, "
            f"proposals={proposal_count}, synthetic={synth_count}"
        )

    print(
        f"Done. sequences={len(sequence_dirs)}, frames={total_frames}, "
        f"proposals={total_proposals}, synthetic={total_synth}"
    )
    print(f"Output root: {config.output_root}")


def build_self_test_dataset(root: Path) -> Path:
    seq_dir = root / "demo_seq"
    seq_dir.mkdir(parents=True, exist_ok=True)
    h, w = 128, 160
    rng = np.random.default_rng(11)

    for idx in range(12):
        base = np.tile(np.linspace(55, 85, w, dtype=np.float32), (h, 1))
        noise = rng.normal(0, 4, size=(h, w)).astype(np.float32)
        frame = base + noise
        cv2.circle(frame, (18 + idx * 8, 38 + idx * 3), 2, 220, -1)
        frame = np.clip(frame, 0, 255).astype(np.uint8)
        cv2.imwrite(str(seq_dir / f"{idx:03d}.png"), frame)

    return root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate masks and proposal metadata from enhanced images, with optional synthetic data generation."
    )
    parser.add_argument("--config", default=str(Path(__file__).with_name("preprocess_config.yaml")), help="Path to YAML config file.")
    parser.add_argument("--input", help="Optional override for enhanced_root (expects sequence folders).")
    parser.add_argument("--output", help="Optional override for output_root.")
    parser.add_argument("--enable-synthetic", action="store_true", help="Enable writing synthetic images under output_root/synthetic.")
    parser.add_argument("--convert-labels", action="store_true", help="Convert proposal JSON to YOLO txt labels after proposal generation.")
    parser.add_argument("--proposal-root", help="Root folder of per-sequence proposal JSON. Default: output_root/proposals.")
    parser.add_argument("--label-root", help="Output root for YOLO txt labels. Default: output_root/labels.")
    parser.add_argument("--class-id", type=int, default=0, help="Class ID written into YOLO labels.")
    parser.add_argument("--score-thresh", type=float, default=5.0, help="Minimum proposal score to keep.")
    parser.add_argument("--max-objects", type=int, default=20, help="Maximum boxes written per frame; <=0 keeps all.")
    parser.add_argument("--self-test", action="store_true", help="Generate a tiny synthetic enhanced set and run this pipeline.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(Path(args.config).resolve())

    if args.self_test:
        temp_root = Path(__file__).resolve().parents[1] / "runs" / "proposal_label_self_test"
        enhanced_root = build_self_test_dataset(temp_root / "enhanced")
        config.enhanced_root = enhanced_root
        config.output_root = temp_root

    if args.input:
        config.enhanced_root = Path(args.input).resolve()
    if args.output:
        config.output_root = Path(args.output).resolve()
    if args.enable_synthetic:
        config.enable_synthetic = True

    if args.convert_labels and not args.proposal_root:
        # Label conversion depends on proposal JSON; force metadata output when using the in-script proposals.
        config.save_metadata = True

    run_pipeline(config)

    if args.convert_labels:
        proposal_root = Path(args.proposal_root).resolve() if args.proposal_root else (config.output_root / "proposals")
        label_root = Path(args.label_root).resolve() if args.label_root else (config.output_root / "labels")
        labels_run(
            proposal_root=proposal_root,
            image_root=config.enhanced_root,
            label_root=label_root,
            class_id=args.class_id,
            score_thresh=args.score_thresh,
            max_objects=args.max_objects,
        )


if __name__ == "__main__":
    main()

