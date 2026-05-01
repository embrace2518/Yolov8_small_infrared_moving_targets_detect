import argparse
import shutil
from pathlib import Path

from utils.logging import get_logger

logger = get_logger(__name__)


def natural_key(path: Path):
    """Sort by numeric stem when possible (e.g., 2.png < 10.png)."""
    stem = path.stem
    return (0, int(stem)) if stem.isdigit() else (1, stem.lower())


def is_image_file(path: Path, exts: set[str]) -> bool:
    return path.is_file() and path.suffix.lower() in exts


def sample_sequence(input_dir: Path, output_dir: Path, step: int, offset: int, exts: set[str], overwrite: bool) -> tuple[int, int]:
    files = [p for p in input_dir.iterdir() if is_image_file(p, exts)]
    files.sort(key=natural_key)

    if not files:
        return 0, 0

    kept = 0
    for idx, src in enumerate(files):
        if idx < offset:
            continue
        if (idx - offset) % step != 0:
            continue

        rel = src.relative_to(input_dir)
        dst = output_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)

        if dst.exists() and not overwrite:
            continue

        shutil.copy2(src, dst)
        kept += 1

    return len(files), kept


def main():
    parser = argparse.ArgumentParser(description="Sample every k-th frame from sequence folders.")
    parser.add_argument("--input", required=True, help="Input root path containing sequence subfolders.")
    parser.add_argument("--output", required=True, help="Output root path for sampled frames.")
    parser.add_argument("--k", type=int, default=5, help="Keep one frame every k frames. Default: 5")
    parser.add_argument("--offset", type=int, default=0, help="Start sampling from this frame index. Default: 0")
    parser.add_argument(
        "--exts",
        default=".png,.jpg,.jpeg,.bmp,.tif,.tiff",
        help="Comma-separated image extensions. Default: .png,.jpg,.jpeg,.bmp,.tif,.tiff",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing files in output.")

    args = parser.parse_args()

    if args.k <= 0:
        raise ValueError("--k must be > 0")
    if args.offset < 0:
        raise ValueError("--offset must be >= 0")

    input_root = Path(args.input).resolve()
    output_root = Path(args.output).resolve()
    exts = {e.strip().lower() if e.strip().startswith(".") else f".{e.strip().lower()}" for e in args.exts.split(",") if e.strip()}

    if not input_root.exists() or not input_root.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_root}")

    total_files = 0
    total_kept = 0
    seq_count = 0

    # Treat each immediate subfolder as one sequence.
    for seq_dir in sorted([p for p in input_root.iterdir() if p.is_dir()], key=lambda p: p.name.lower()):
        seq_count += 1
        out_seq_dir = output_root / seq_dir.name
        count, kept = sample_sequence(seq_dir, out_seq_dir, args.k, args.offset, exts, args.overwrite)
        total_files += count
        total_kept += kept
        logger.info("%s: %s/%s kept", seq_dir.name, kept, count)

    if seq_count == 0:
        logger.warning("No sequence folders found under input root.")
    else:
        ratio = (100.0 * total_kept / total_files) if total_files else 0.0
        logger.info("Done. sequences=%s, kept=%s/%s (%.2f%%)", seq_count, total_kept, total_files, ratio)
        logger.info("Output: %s", output_root)


if __name__ == "__main__":
    main()

