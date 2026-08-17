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


def _leaf_isolation_panel(image, detections):
    """Colour-tinted overlay, one distinct colour + ID per detected leaf —
    the Track B analogue of Track A's watershed leaf-count panel. Track A
    needs watershed because it detects one box per whole plant and has to
    split it into leaves after the fact; Track B detects each leaf as its
    own box directly, so this just tints what the model already separated.
    It is NOT a pixel-accurate segmentation mask (Track B has no per-pixel
    leaf boundaries, only boxes) — tinting the box region is an honest
    visualization of "which detection is which", not a claim of instance
    segmentation."""
    overlay = image.copy()
    rng = np.random.RandomState(42)
    for i, det in enumerate(detections, start=1):
        color = tuple(int(c) for c in rng.randint(60, 255, 3))
        x1, y1, x2, y2 = [int(v) for v in det.box]
        x1, y1 = max(x1, 0), max(y1, 0)
        x2, y2 = min(x2, image.shape[1]), min(y2, image.shape[0])
        if x2 <= x1 or y2 <= y1:
            continue
        region = overlay[y1:y2, x1:x2]
        tint = np.full_like(region, color)
        overlay[y1:y2, x1:x2] = cv2.addWeighted(region, 0.55, tint, 0.45, 0)
        cv2.putText(overlay, str(i), (x1 + 3, y1 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    return overlay


def plot_leaf_pipeline_grid(
    image_paths,
    weights_path,
    out_path,
    row_labels: list = None,
    imgsz: int = 640,
    conf: float = 0.25,
    class_names: list = None,
    augment: bool = False,
    show_track_a_mask_comparison: bool = False,
) -> Path:
    """Track B's equivalent of plot_pipeline_grid: Original Input | Model
    Input (letterboxed to imgsz) | Final Detection | Detected Leaves
    (isolated). No masking stage by default — unlike Track A, leaf
    detection trains and infers on plain RGB, so showing an HSV mask step
    as if it were part of Track B's pipeline would misrepresent it. The
    isolated-leaves column plays the same visual role as Track A's
    watershed leaf-count column, but built from Track B's own per-leaf
    boxes rather than a watershed split — Track B doesn't need watershed
    since it detects each leaf directly (see _leaf_isolation_panel).

    show_track_a_mask_comparison=True inserts two extra columns —
    "HSV Green Mask" and "Masked View" — computed with Track A's own
    green_mask()/apply_mask_soft(), explicitly titled as Track A's
    technique applied for comparison, not part of Track B's real
    pipeline. This isn't decoration: it's the actual evidence for why
    Track B doesn't mask — dry/brown fallen leaves aren't green, so the
    mask discards most of the leaf, which the green-pixel-% annotation
    under that column makes concrete per image.

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

    stage_titles = ["Original Input"]
    if show_track_a_mask_comparison:
        stage_titles += ["HSV Green Mask\n(Track A technique — for comparison)",
                          "Masked View\n(Track A technique — for comparison)"]
    stage_titles += [f"Model Input ({imgsz}×{imgsz})", "Final Detection", "Detected Leaves (isolated)"]
    n_cols = len(stage_titles)

    n = len(image_paths)
    if row_labels is not None and len(row_labels) != n:
        raise ValueError("row_labels must be the same length as image_paths")

    fig, axes = plt.subplots(n, n_cols, figsize=(4.25 * n_cols, 4.3 * n), squeeze=False)

    for row, img_path in enumerate(image_paths):
        image = cv2.imread(str(img_path))
        if image is None:
            raise FileNotFoundError(img_path)

        stage_images = [cv2.cvtColor(image, cv2.COLOR_BGR2RGB)]
        green_pct = None
        if show_track_a_mask_comparison:
            mask = green_mask(image)
            masked_view = apply_mask_soft(image, mask)
            green_pct = 100.0 * (mask > 0).mean()
            stage_images += [mask, cv2.cvtColor(masked_view, cv2.COLOR_BGR2RGB)]

        model_input = _letterbox(image, imgsz)
        detections = _detect_leaf(model, image, conf, augment)
        detected = draw_detections(image, detections, class_names)
        isolated = _leaf_isolation_panel(image, detections)
        stage_images += [
            cv2.cvtColor(model_input, cv2.COLOR_BGR2RGB),
            cv2.cvtColor(detected, cv2.COLOR_BGR2RGB),
            cv2.cvtColor(isolated, cv2.COLOR_BGR2RGB),
        ]
        row_label = row_labels[row] if row_labels else Path(img_path).stem

        for col, (title, img) in enumerate(zip(stage_titles, stage_images)):
            ax = axes[row][col]
            ax.imshow(img, cmap="gray" if img.ndim == 2 else None)
            if row == 0:
                ax.set_title(title, fontsize=11)
            if col == 0:
                ax.set_ylabel(row_label, fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
            if show_track_a_mask_comparison and col == 1:
                ax.set_xlabel(f"only {green_pct:.0f}% green", fontsize=9)
            if col == n_cols - 2 or col == n_cols - 1:
                ax.set_xlabel(f"{len(detections)} leaf/leaves detected", fontsize=9)

    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path
