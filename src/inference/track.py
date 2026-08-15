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


class TrackSmoother:
    """Exponential moving average on each tracked box's center point, keyed
    by track id — a deliberately simpler alternative to a Kalman filter.
    A Kalman filter models velocity/acceleration and needs process-noise
    tuning; EMA needs one parameter and no motion model, and achieves the
    same practical goal here (smooth, non-jittery coordinates for a
    downstream robot controller) with a fraction of the code. Swap in a
    Kalman filter later only if velocity/acceleration estimates are
    actually needed, not just smoothed position."""

    def __init__(self, alpha: float = 0.4):
        self.alpha = alpha
        self._state: dict[int, tuple] = {}

    def smooth(self, track_id: int, x: float, y: float) -> tuple:
        prev = self._state.get(track_id)
        if prev is None:
            smoothed = (x, y)
        else:
            a = self.alpha
            smoothed = (a * x + (1 - a) * prev[0], a * y + (1 - a) * prev[1])
        self._state[track_id] = smoothed
        return smoothed

    def prune(self, active_ids: set):
        """Drop state for ids no longer present, so an id reused after a
        long gap doesn't inherit a stale position."""
        for tid in [t for t in self._state if t not in active_ids]:
            del self._state[tid]


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


def predict_image_response(
    weights_path: str,
    image_path: str,
    conf: float = 0.25,
    class_names: list = None,
    augment: bool = False,
) -> dict:
    """One image in, one Section-20-shaped response out. augment=True
    enables test-time augmentation (slower, more thorough — see
    predict.py's predict_image for the same tradeoff note)."""
    model = YOLO(weights_path)
    class_names = class_names or ["leaf"]
    results = model.predict(str(image_path), conf=conf, augment=augment, verbose=False)[0]

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
    smooth: bool = False,
    smooth_alpha: float = 0.4,
) -> Iterator[dict]:
    """Yields one dict per frame, matching spec Section 7's tracked-leaf
    shape (id persists across frames for the same physical leaf) plus
    frame-level stats (spec Section 6's live-mode display fields).
    smooth=False by default so existing captured output (e.g. an already
    -reported sample frame) stays reproducible; set True to apply EMA
    smoothing to center_pixel coordinates — see TrackSmoother."""
    model = YOLO(weights_path)
    class_names = class_names or ["leaf"]
    smoother = TrackSmoother(smooth_alpha) if smooth else None

    for frame_idx, result in enumerate(model.track(source, conf=conf, tracker=tracker, persist=True, stream=True, verbose=False)):
        start = time.perf_counter()
        tracks = []
        confidences = []
        active_ids = set()
        if result.boxes.id is not None:
            for box, track_id in zip(result.boxes, result.boxes.id):
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf_val = float(box.conf[0])
                confidences.append(conf_val)
                cls = int(box.cls[0])
                name = class_names[cls] if cls < len(class_names) else str(cls)
                tid = int(track_id)
                active_ids.add(tid)

                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                if smoother is not None:
                    cx, cy = smoother.smooth(tid, cx, cy)

                tracks.append({
                    "id": tid,
                    "class": name,
                    "confidence": round(conf_val, 4),
                    "center_pixel": {"x": round(cx, 1), "y": round(cy, 1)},
                    "status": "detected",
                })

        if smoother is not None:
            smoother.prune(active_ids)

        elapsed = max(time.perf_counter() - start, 1e-6)
        yield {
            "frame": frame_idx,
            "leaves_detected": len(tracks),
            "average_confidence": round(sum(confidences) / len(confidences), 4) if confidences else 0.0,
            "fps": round(1.0 / elapsed, 1),
            "status": "READY",
            "smoothed": smooth,
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
