"""Run the vegetation model and a separate waste-detection model on the same
image and merge both detection sets into one annotated view. Two independent
models rather than one retrained multi-class model, so the validated
vegetation ablation results are never disturbed by changes on the waste
side (see README 'Waste detection' section for the reasoning)."""

from dataclasses import dataclass
from pathlib import Path

import cv2
from ultralytics import YOLO

from .predict import Detection, draw_detections
from ..preprocessing.hsv_mask import MaskConfig, green_pixel_ratio

VEG_COLOR = (60, 200, 60)     # green — matches predict.BOX_COLOR
WASTE_COLOR = (40, 40, 230)   # red (BGR)


@dataclass
class CombinedResult:
    image_path: Path
    vegetation: list
    waste: list


def _run_model(model, image, conf):
    results = model.predict(image, conf=conf, verbose=False)[0]
    return [
        Detection(box=tuple(box.xyxy[0].tolist()), conf=float(box.conf[0]), cls=int(box.cls[0]))
        for box in results.boxes
    ]


def predict_combined(
    image_path: str,
    veg_weights: str,
    waste_weights: str,
    veg_conf: float = 0.25,
    waste_conf: float = 0.25,
    veg_class_names: list = None,
    waste_class_names: list = None,
    veg_green_ratio_threshold: float = None,
    mask_config: MaskConfig = None,
):
    """Returns (annotated_image_bgr, CombinedResult). Vegetation boxes drawn
    green, waste boxes drawn red — same image, two independent model passes."""
    veg_model = YOLO(veg_weights)
    waste_model = YOLO(waste_weights)
    veg_class_names = veg_class_names or ["green_vegetation"]
    waste_class_names = waste_class_names or []

    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(image_path)

    veg_detections = _run_model(veg_model, image, veg_conf)
    if veg_green_ratio_threshold is not None:
        veg_detections = [
            d for d in veg_detections
            if green_pixel_ratio(image, d.box, mask_config) >= veg_green_ratio_threshold
        ]
    waste_detections = _run_model(waste_model, image, waste_conf)

    annotated = draw_detections(image, veg_detections, veg_class_names, VEG_COLOR)
    annotated = draw_detections(annotated, waste_detections, waste_class_names, WASTE_COLOR)

    return annotated, CombinedResult(Path(image_path), veg_detections, waste_detections)


def predict_combined_and_save(image_path: str, veg_weights: str, waste_weights: str, out_path: str, **kwargs) -> Path:
    annotated, result = predict_combined(image_path, veg_weights, waste_weights, **kwargs)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), annotated)
    return out_path
