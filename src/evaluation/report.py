"""Combined original -> mask -> masked-view -> final-detection figure — the
single artifact that tells the whole pipeline story per sample, for the
report/competition writeup."""

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from ultralytics import YOLO

from ..analysis.leaf_counter import count_leaves_in_regions
from ..inference.predict import Detection, draw_detections
from ..preprocessing.hsv_mask import MaskConfig, apply_mask_soft, green_mask, green_pixel_ratio

BASE_STAGE_TITLES = ("Original", "Green Mask", "Masked (training view)", "Final Detection")
LEAF_STAGE_TITLE = "Leaf Count (watershed)"


def _detect(model, image, conf, green_ratio_threshold, mask_config, augment=False):
    results = model.predict(image, conf=conf, augment=augment, verbose=False)[0]
    detections = []
    for box in results.boxes:
        xyxy = tuple(box.xyxy[0].tolist())
        ratio = green_pixel_ratio(image, xyxy, mask_config) if green_ratio_threshold is not None else None
        if green_ratio_threshold is not None and ratio < green_ratio_threshold:
            continue
        detections.append(Detection(box=xyxy, conf=float(box.conf[0]), cls=int(box.cls[0]), green_ratio=ratio))
    return detections


def plot_pipeline_grid(
    image_paths,
    weights_path,
    out_path,
    mask_config: MaskConfig = None,
    conf: float = 0.25,
    green_ratio_threshold: float = None,
    class_names: list = None,
    show_leaf_count: bool = True,
    leaf_min_area: int = 120,
    leaf_fg_ratio: float = 0.35,
    augment: bool = False,
) -> Path:
    """Save a len(image_paths) x (4 or 5) grid: Original | Green Mask |
    Masked (training view) | Final Detection | [Leaf Count]. Detection runs
    on the original, unmasked image — masking is training-time only. Leaf
    counting is a classical watershed split on the mask within each detected
    box (see src/analysis/leaf_counter.py) — an estimate, not ground truth;
    it under-counts leaves that fully overlap in the 2D projection."""
    model = YOLO(weights_path)
    class_names = class_names or ["green_vegetation"]
    stage_titles = BASE_STAGE_TITLES + ((LEAF_STAGE_TITLE,) if show_leaf_count else ())
    n_cols = len(stage_titles)

    n = len(image_paths)
    fig, axes = plt.subplots(n, n_cols, figsize=(4 * n_cols, 4 * n), squeeze=False)

    for row, img_path in enumerate(image_paths):
        image = cv2.imread(str(img_path))
        if image is None:
            raise FileNotFoundError(img_path)

        mask = green_mask(image, mask_config)
        masked = apply_mask_soft(image, mask)
        detections = _detect(model, image, conf, green_ratio_threshold, mask_config, augment)
        detected = draw_detections(image, detections, class_names)

        stage_images = [
            cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
            mask,
            cv2.cvtColor(masked, cv2.COLOR_BGR2RGB),
            cv2.cvtColor(detected, cv2.COLOR_BGR2RGB),
        ]

        leaf_total = None
        if show_leaf_count:
            boxes = [d.box for d in detections]
            _, leaf_overlay, leaf_total = count_leaves_in_regions(
                image, boxes, mask_config, leaf_min_area, leaf_fg_ratio
            )
            stage_images.append(cv2.cvtColor(leaf_overlay, cv2.COLOR_BGR2RGB))

        for col, (title, img) in enumerate(zip(stage_titles, stage_images)):
            ax = axes[row][col]
            ax.imshow(img, cmap="gray" if img.ndim == 2 else None)
            if row == 0:
                ax.set_title(title, fontsize=12)
            if col == 0:
                ax.set_ylabel(Path(img_path).stem, fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
            if col == 3:
                ax.set_xlabel(f"{len(detections)} detection(s)", fontsize=9)
            elif show_leaf_count and col == 4:
                ax.set_xlabel(f"{leaf_total} leaf/leaves", fontsize=9)

    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def _letterbox(image, new_size: int = 640, color: tuple = (114, 114, 114)):
    """Resize-and-pad to a square new_size x new_size canvas, preserving
    aspect ratio — the same transform YOLO applies internally before an
    image reaches the network. Shown as its own pipeline stage because it's
    literally what "entering the model" means for a YOLO detector, not
    just a resize for display."""
    h, w = image.shape[:2]
    scale = min(new_size / h, new_size / w)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((new_size, new_size, 3), color, dtype=np.uint8)
    top, left = (new_size - nh) // 2, (new_size - nw) // 2
    canvas[top:top + nh, left:left + nw] = resized
    return canvas


def _detect_leaf(model, image, conf, augment=False):
    results = model.predict(image, conf=conf, augment=augment, verbose=False)[0]
    return [
        Detection(box=tuple(b.xyxy[0].tolist()), conf=float(b.conf[0]), cls=int(b.cls[0]))
        for b in results.boxes
    ]


def plot_leaf_pipeline_grid(
    image_paths,
    weights_path,
    out_path,
    row_labels: list = None,
    imgsz: int = 640,
    conf: float = 0.25,
    class_names: list = None,
    augment: bool = False,
) -> Path:
    """Track B's equivalent of plot_pipeline_grid: Original Input | Model
    Input (letterboxed to imgsz) | Final Detection. No masking stage here —
    unlike Track A, leaf detection trains and infers on plain RGB, so
    showing an HSV mask step would misrepresent the actual pipeline.

    Runs inference only against an already-trained checkpoint (e.g.
    leaf_v3_weights) — no training happens here.

    row_labels: pass explicit labels (e.g. 'REAL PHOTO — foo.jpg' vs
    'SYNTHETIC — bar.jpg') so the figure can never blur a synthetic/AI
    image into looking like a real one — defaults to the filename stem,
    which does NOT indicate provenance and should be overridden when that
    distinction matters for the reader.
    """
    model = YOLO(weights_path)
    class_names = class_names or ["leaf"]
    stage_titles = ("Original Input", f"Model Input ({imgsz}×{imgsz})", "Final Detection")
    n = len(image_paths)
    if row_labels is not None and len(row_labels) != n:
        raise ValueError("row_labels must be the same length as image_paths")

    fig, axes = plt.subplots(n, 3, figsize=(13, 4.3 * n), squeeze=False)

    for row, img_path in enumerate(image_paths):
        image = cv2.imread(str(img_path))
        if image is None:
            raise FileNotFoundError(img_path)

        model_input = _letterbox(image, imgsz)
        detections = _detect_leaf(model, image, conf, augment)
        detected = draw_detections(image, detections, class_names)

        stage_images = [
            cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
            cv2.cvtColor(model_input, cv2.COLOR_BGR2RGB),
            cv2.cvtColor(detected, cv2.COLOR_BGR2RGB),
        ]
        row_label = row_labels[row] if row_labels else Path(img_path).stem

        for col, (title, img) in enumerate(zip(stage_titles, stage_images)):
            ax = axes[row][col]
            ax.imshow(img)
            if row == 0:
                ax.set_title(title, fontsize=12)
            if col == 0:
                ax.set_ylabel(row_label, fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
            if col == 2:
                ax.set_xlabel(f"{len(detections)} leaf/leaves detected", fontsize=9)

    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path
