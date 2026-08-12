"""Combined original -> mask -> masked-view -> final-detection figure — the
single artifact that tells the whole pipeline story per sample, for the
report/competition writeup."""

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO

from ..analysis.leaf_counter import count_leaves_in_regions
from ..inference.predict import Detection, draw_detections
from ..preprocessing.hsv_mask import MaskConfig, apply_mask_soft, green_mask, green_pixel_ratio

BASE_STAGE_TITLES = ("Original", "Green Mask", "Masked (training view)", "Final Detection")
LEAF_STAGE_TITLE = "Leaf Count (watershed)"


def _detect(model, image, conf, green_ratio_threshold, mask_config):
    results = model.predict(image, conf=conf, verbose=False)[0]
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
        detections = _detect(model, image, conf, green_ratio_threshold, mask_config)
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
