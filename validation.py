from __future__ import annotations

# Validation and Scoring tool for the spotGEO competition on kelvins.esa.int.

# Imports
import json
import numpy as np
from collections import defaultdict
from pathlib import Path

from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

# Constants
min_seq_number = 1
max_seq_number = 3  # 5120
frames_per_sequence = 1000
img_width  = 639.5
img_height = 479.5
min_size = -0.5
max_number_of_objects = 30

# Helper functions
def flat_to_hierarchical(labels: list[dict]) -> dict[int, dict[int, np.ndarray]]:
    """ Transforms a flat array of json-objects to a hierarchical python dict, indexed by
        sequence number and frame id. """
    seqs = dict()
    for label in labels:
        seq_id = label['sequence_id']
        frame_id = label['frame']
        coords = label['object_coords']

        if seq_id not in seqs.keys():
            seqs[seq_id] = defaultdict(dict)
        seqs[seq_id][frame_id] = np.array(coords)

    return seqs


def score_frame(
    X: np.ndarray,
    Y: np.ndarray,
    tau: float = 10,
    eps: float = 3,
) -> tuple[int, int, int, float]:
    """ Scoring Prediction X on ground-truth Y by linear assignment. """
    if len(X) == 0 and len(Y) == 0:
        # no objects, no predictions means perfect score
        TP, FN, FP, sse = 0, 0, 0, 0
    elif len(X) == 0 and len(Y) > 0:
        # no predictions but objects means false negatives
        TP, FN, FP, sse = 0, len(Y), 0, len(Y) * tau**2
    elif len(X) > 0 and len(Y) == 0:
        # predictions but no objects means false positives
        TP, FN, FP, sse = 0, 0, len(X), len(X) * tau**2
    else:
        # compute Euclidean distances between prediction and ground truth
        D = cdist(X, Y)

        # truncate distances that violate the threshold
        D[D > tau] = 1000

        # compute matching by solving linear assignment problem
        row_ind, col_ind = linear_sum_assignment(D)
        matching = D[row_ind, col_ind]

        # true positives are matches within the threshold
        TP = sum(matching <= tau)

        # false negatives are missed ground truth points or matchings that violate the threshold
        FN = len(Y) - len(row_ind) + sum(matching > tau)

        # false positives are missing predictions or matchings that violate the threshold
        FP = len(X) - len(row_ind) + sum(matching > tau)

        # compute truncated regression error
        tp_distances = matching[matching < tau]
        # truncation
        tp_distances[tp_distances < eps] = 0
        # squared error with constant punishment for false negatives and true positives
        sse = sum(tp_distances) + (FN + FP) * tau**2

    return TP, FN, FP, sse


def score_sequence(
    X: dict[int, np.ndarray],
    Y: dict[int, np.ndarray],
    tau: float = 10,
    eps: float = 3,
) -> tuple[int, int, int, float]:
    # 部分预测模式：
    # - Y 有而 X 无 -> 按空预测计 FN
    # - X 有而 Y 无 -> 按空标注计 FP
    frame_ids = set(X.keys()) | set(Y.keys())

    frame_scores = [
        score_frame(
            X.get(k, np.zeros((0, 2), dtype=float)),
            Y.get(k, np.zeros((0, 2), dtype=float)),
            tau=tau,
            eps=eps,
        )
        for k in frame_ids
    ]
    TP = sum(x[0] for x in frame_scores)
    FN = sum(x[1] for x in frame_scores)
    FP = sum(x[2] for x in frame_scores)
    sse = sum(x[3] for x in frame_scores)

    mse = 0 if (TP + FN + FP) == 0 else sse / (TP + FN + FP)
    return TP, FN, FP, mse


def score_sequences(
    X: dict,
    Y: dict,
    tau: float = 10,
    eps: float = 3,
    taboolist: list | None = None,
) -> tuple[float, float, float, float]:
    """ scores a complete submission except sequence_ids that are listed
        in the taboolist. """
    # check that each sequence has been predicted
    #assert set(X.keys()) == set(Y.keys())

    # A 模式：标签全集是评估主集合；缺失预测自动计 FN。
    # 同时保留预测中额外序列，按 FP 惩罚。
    taboo = set(taboolist or [])
    identifiers = (set(X.keys()) | set(Y.keys())) - taboo

    # compute individual sequence scores
    seq_scores = [
        score_sequence(
            X.get(k, defaultdict(dict)),
            Y.get(k, defaultdict(dict)),
            tau=tau,
            eps=eps,
        )
        for k in identifiers
    ]
    TP = sum(x[0] for x in seq_scores)
    FN = sum(x[1] for x in seq_scores)
    FP = sum(x[2] for x in seq_scores)
    mse = sum(x[3] for x in seq_scores)

    precision = 0.0 if (TP + FP) == 0 else TP / (TP + FP)
    recall = 0.0 if (TP + FN) == 0 else TP / (TP + FN)
    F1 = 0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)

    return precision, recall, F1, mse


def compute_score(predictions: str | Path, labels: str | Path) -> tuple[float, float]:
    """ Scores a submission `predictions` against ground-truth `labels`. Does
    not perform any validation and expects `predictions` and `labels` to be
    valid paths to .json-files. """
    with open(predictions, 'rt') as fp:
        predictions_h = flat_to_hierarchical(json.load(fp))

    with open(labels, 'rt') as fp:
        labels_h = flat_to_hierarchical(json.load(fp))

    precision, recall, F1, mse = score_sequences(predictions_h, labels_h)

    return (1 - F1, mse)


def validate_json(labels: list, require_full_coverage: bool = False) -> bool:
    """ Validates whether `labels` follow the required formats to be accepted
        for computing a score. Automatically adapts to generated JSON format
        without rigid schema restrictions that break on arbitrary sequence IDs. """
    if not isinstance(labels, list):
        raise ValueError("JSON root must be an array (list).")

    identifiers = []
    for label in labels:
        if not all(k in label for k in ["sequence_id", "frame", "num_objects", "object_coords"]):
            raise ValueError(f"Missing required keys in label: {label}")

        identifiers.append((str(label['sequence_id']), str(label['frame'])))

        if len(label.get('object_coords', [])) != label.get('num_objects', 0):
            raise ValueError(
                f"Error. indicated num_objects={label.get('num_objects')} "
                f"but gave {len(label.get('object_coords', []))} coords in seq {label['sequence_id']} frame {label['frame']}."
            )

    if len(set(identifiers)) != len(identifiers):
        raise ValueError('Error. You have duplicates in your submission. Make sure each combination of sequence_id and frame is unique.')

    return True


def _to_int_or_self(value: str) -> int | str:
    return int(value) if value.lstrip("-").isdigit() else value


def _resolve_sources(sources: str | Path | list[str | Path]) -> list[Path]:
    """
    统一解析源参数（支持单个或列表），返回所有图像路径。

    Args:
        sources: 一个或多个目录/文件的路径。

    Returns:
        所有图像文件的路径列表（去重、排序）。
    """
    def _load_image_paths(source: str | Path) -> list[Path]:
        source_path = Path(source)
        if source_path.is_dir():
            paths = sorted(p for p in source_path.rglob("*")
                           if p.suffix.lower() in ('.png', '.jpg', '.jpeg'))
        elif source_path.is_file():
            paths = [source_path]
        else:
            paths = []
        return paths

    if isinstance(sources, (str, Path)):
        sources = [sources]

    all_paths: list[Path] = []
    seen: set[Path] = set()
    for src in sources:
        paths = _load_image_paths(src)
        for p in paths:
            resolved = p.resolve()
            if resolved not in seen:
                seen.add(resolved)
                all_paths.append(p)

    return sorted(all_paths)
