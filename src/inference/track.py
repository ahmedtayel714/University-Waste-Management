"""Video tracking + the spec's JSON output schemas. Tracking uses
Ultralytics' native ByteTrack/BoT-SORT support (model.track()) rather than a
custom tracker — the spec only requires stable per-leaf IDs across frames,
which the built-in tracker already provides, so there's nothing to gain from
reimplementing it. This module just formats results to the interface
contract the (future, currently absent) robot-side code expects."""

import time
from pathlib import Path
from typing import Iterator

from ultralytics import YOLO


def detection_to_schema(box_xyxy, conf: float, cls: int, class_names: list) -> dict:
    """Matches spec Section 3's per-detection JSON."""
    x1, y1, x2, y2 = box_xyxy
    name = class_names[cls] if cls < len(class_names) else str(cls)
    return {
        "class": name,
        "confidence": round(float(conf), 4),
        "bbox": {"x1": round(x1, 1), "y1": round(y1, 1), "x2": round(x2, 1), "y2": round(y2, 1)},
        "center": {"x": round((x1 + x2) / 2, 1), "y": round((y1 + y2) / 2, 1)},
        "width": round(x2 - x1, 1),
        "height": round(y2 - y1, 1),
    }


def detections_response(detections: list, image_width: int, image_height: int) -> dict:
    """Matches spec Section 20's /detections API response shape."""
    return {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "image_width": image_width,
        "image_height": image_height,
        "leaf_count": len(detections),
        "detections": [{"id": i + 1, **d} for i, d in enumerate(detections)],
    }


def predict_image_response(weights_path: str, image_path: str, conf: float = 0.25, class_names: list = None) -> dict:
    """One image in, one Section-20-shaped response out."""
    model = YOLO(weights_path)
    class_names = class_names or ["leaf"]
    results = model.predict(str(image_path), conf=conf, verbose=False)[0]

    detections = [
        detection_to_schema(box.xyxy[0].tolist(), box.conf[0], int(box.cls[0]), class_names)
        for box in results.boxes
    ]
    h, w = results.orig_shape
    return detections_response(detections, w, h)


def track_video(
    weights_path: str,
    source: str,
    conf: float = 0.25,
    tracker: str = "bytetrack.yaml",
    class_names: list = None,
) -> Iterator[dict]:
    """Yields one dict per frame, matching spec Section 7's tracked-leaf
    shape (id persists across frames for the same physical leaf) plus
    frame-level stats (spec Section 6's live-mode display fields)."""
    model = YOLO(weights_path)
    class_names = class_names or ["leaf"]

    for frame_idx, result in enumerate(model.track(source, conf=conf, tracker=tracker, persist=True, stream=True, verbose=False)):
        start = time.perf_counter()
        tracks = []
        confidences = []
        if result.boxes.id is not None:
            for box, track_id in zip(result.boxes, result.boxes.id):
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf_val = float(box.conf[0])
                confidences.append(conf_val)
                cls = int(box.cls[0])
                name = class_names[cls] if cls < len(class_names) else str(cls)
                tracks.append({
                    "id": int(track_id),
                    "class": name,
                    "confidence": round(conf_val, 4),
                    "center_pixel": {
                        "x": round((x1 + x2) / 2, 1),
                        "y": round((y1 + y2) / 2, 1),
                    },
                    "status": "detected",
                })

        elapsed = max(time.perf_counter() - start, 1e-6)
        yield {
            "frame": frame_idx,
            "leaves_detected": len(tracks),
            "average_confidence": round(sum(confidences) / len(confidences), 4) if confidences else 0.0,
            "fps": round(1.0 / elapsed, 1),
            "status": "READY",
            "tracks": tracks,
        }


def track_video_to_json(weights_path: str, source: str, out_path: str, **kwargs) -> Path:
    """Convenience: run track_video over a whole video and dump the
    per-frame records to a single JSON array file."""
    import json

    frames = list(track_video(weights_path, source, **kwargs))
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(frames, indent=2))
    return out_path
