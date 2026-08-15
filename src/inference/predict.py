"""Inference on original, unmasked RGB images. A model trained on masked
imagery still runs inference on raw images at deployment time — masking is a
training-time noise filter, not a runtime dependency — with an optional
green-ratio post-filter to cut background false positives."""

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from ..preprocessing.hsv_mask import MaskConfig, green_pixel_ratio

BOX_COLOR = (60, 200, 60)
TEXT_COLOR = (255, 255, 255)


@dataclass
class Detection:
    box: tuple  # x1, y1, x2, y2
    conf: float
    cls: int
    green_ratio: float = None


def predict_image(
    weights_path: str,
    image_path: str,
    conf: float = 0.25,
    green_ratio_threshold: float = None,
    mask_config: MaskConfig = None,
    augment: bool = False,
) -> tuple[np.ndarray, list[Detection]]:
    """Run detection on one image. If green_ratio_threshold is set, detections
    whose box interior is less than that fraction green (in HSV space) are
    dropped as likely soil/straw false positives. augment=True enables
    Ultralytics' test-time augmentation (multi-scale + flip, averaged) —
    catches small/marginal detections at the cost of ~2-3x slower inference,
    so it's off by default and meant for final evaluation, not live video."""
    model = YOLO(weights_path)
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(image_path)

    results = model.predict(image, conf=conf, augment=augment, verbose=False)[0]

    detections = []
    for box in results.boxes:
        xyxy = tuple(box.xyxy[0].tolist())
        ratio = green_pixel_ratio(image, xyxy, mask_config) if green_ratio_threshold is not None else None
        if green_ratio_threshold is not None and ratio < green_ratio_threshold:
            continue
        detections.append(Detection(box=xyxy, conf=float(box.conf[0]), cls=int(box.cls[0]), green_ratio=ratio))

    return image, detections


def draw_detections(
    image: np.ndarray,
    detections: list[Detection],
    class_names: list[str] = None,
    color: tuple = BOX_COLOR,
) -> np.ndarray:
    class_names = class_names or ["green_vegetation"]
    annotated = image.copy()
    for det in detections:
        x1, y1, x2, y2 = [int(v) for v in det.box]
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        name = class_names[det.cls] if det.cls < len(class_names) else str(det.cls)
        label = f"{name} {det.conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(annotated, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(annotated, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, TEXT_COLOR, 1)
    return annotated


def predict_and_save(
    weights_path: str,
    image_path: str,
    out_path: str,
    conf: float = 0.25,
    green_ratio_threshold: float = None,
    mask_config: MaskConfig = None,
    class_names: list[str] = None,
    augment: bool = False,
) -> Path:
    image, detections = predict_image(weights_path, image_path, conf, green_ratio_threshold, mask_config, augment)
    annotated = draw_detections(image, detections, class_names)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), annotated)
    return out_path
