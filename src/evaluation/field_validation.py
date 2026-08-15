"""Real-world validation harness for a folder of real photos, run against a
trained model. If ground-truth YOLO labels are provided, computes true
mAP/precision/recall exactly like training-time validation; otherwise
reports detection-rate and confidence statistics and saves annotated
snapshots for manual review. Ready to run the moment real photos exist —
this module can't fabricate the photos themselves."""

import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import yaml
from ultralytics import YOLO

from ..inference.predict import Detection, draw_detections

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def run_field_validation(
    weights_path: str,
    field_dir: Path,
    out_dir: Path,
    conf: float = 0.25,
    class_names: list = None,
    imgsz: int = 640,
) -> dict:
    """field_dir must contain an images/ subfolder (flat, any real photos).
    If it also contains a labels/ subfolder with matching YOLO .txt files,
    real mAP/precision/recall are computed via model.val(); otherwise this
    reports detection-rate and confidence statistics only — still useful
    (catches 'the model detects nothing on real photos' early) but is not
    a substitute for measured accuracy."""
    field_dir = Path(field_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    class_names = class_names or ["leaf"]

    images_dir = field_dir / "images"
    labels_dir = field_dir / "labels"
    if not images_dir.exists():
        raise FileNotFoundError(f"{images_dir} does not exist — field_dir must contain an images/ subfolder")

    image_paths = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    if not image_paths:
        raise FileNotFoundError(f"No images found in {images_dir}")

    model = YOLO(weights_path)
    result = {"n_images": len(image_paths), "has_labels": labels_dir.exists()}

    detection_counts = []
    all_confidences = []
    annotated_dir = out_dir / "annotated"
    annotated_dir.mkdir(exist_ok=True)

    for img_path in image_paths:
        image = cv2.imread(str(img_path))
        if image is None:
            continue
        preds = model.predict(image, conf=conf, verbose=False)[0]
        detections = [
            Detection(box=tuple(b.xyxy[0].tolist()), conf=float(b.conf[0]), cls=int(b.cls[0]))
            for b in preds.boxes
        ]
        detection_counts.append(len(detections))
        all_confidences.extend(d.conf for d in detections)

        annotated = draw_detections(image, detections, class_names)
        cv2.imwrite(str(annotated_dir / img_path.name), annotated)

    result["images_with_zero_detections"] = sum(1 for c in detection_counts if c == 0)
    result["mean_detections_per_image"] = round(float(np.mean(detection_counts)), 2) if detection_counts else 0.0
    result["mean_confidence"] = round(float(np.mean(all_confidences)), 4) if all_confidences else None
    result["median_confidence"] = round(float(np.median(all_confidences)), 4) if all_confidences else None

    if all_confidences:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.hist(all_confidences, bins=20, color="#3F5A43", edgecolor="white")
        ax.set_xlabel("confidence")
        ax.set_ylabel("count")
        ax.set_title(f"Field validation — confidence distribution ({len(image_paths)} real photos)")
        fig.tight_layout()
        fig.savefig(out_dir / "confidence_histogram.png", dpi=150)
        plt.close(fig)

    if labels_dir.exists():
        data_yaml_path = out_dir / "field_data.yaml"
        data_yaml_path.write_text(yaml.safe_dump({
            "path": str(field_dir.resolve()),
            "train": "images",
            "val": "images",
            "nc": len(class_names),
            "names": class_names,
        }, sort_keys=False))
        val_results = model.val(data=str(data_yaml_path), imgsz=imgsz, plots=True, split="val")
        result["precision"] = float(val_results.box.mp)
        result["recall"] = float(val_results.box.mr)
        result["mAP50"] = float(val_results.box.map50)
        result["mAP50-95"] = float(val_results.box.map)

    (out_dir / "field_validation_summary.json").write_text(json.dumps(result, indent=2))
    return result
