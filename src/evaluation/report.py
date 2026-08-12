"""Combined original -> mask -> masked-view -> final-detection figure — the
single artifact that tells the whole pipeline story per sample, for the
report/competition writeup."""

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO

from ..inference.predict import Detection, draw_detections
from ..preprocessing.hsv_mask import MaskConfig, apply_mask_soft, green_mask, green_pixel_ratio

STAGE_TITLES = ("Original", "Green Mask", "Masked (training view)", "Final Detection")


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
) -> Path:
    """Save a len(image_paths) x 4 grid: Original | Green Mask | Masked
    (training view) | Final Detection. Detection runs on the original,
    unmasked image — masking is training-time only."""
    model = YOLO(weights_path)
    class_names = class_names or ["green_vegetation"]

    n = len(image_paths)
    fig, axes = plt.subplots(n, 4, figsize=(16, 4 * n), squeeze=False)

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
        for col, (title, img) in enumerate(zip(STAGE_TITLES, stage_images)):
            ax = axes[row][col]
            ax.imshow(img, cmap="gray" if img.ndim == 2 else None)
            if row == 0:
                ax.set_title(title, fontsize=12)
            if col == 0:
                ax.set_ylabel(Path(img_path).stem, fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
            det_count = len(detections) if col == 3 else None
            if det_count is not None:
                ax.set_xlabel(f"{det_count} detection(s)", fontsize=9)

    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path
