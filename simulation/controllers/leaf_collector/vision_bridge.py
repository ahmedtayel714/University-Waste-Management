"""Bridges a Webots camera frame to the project's real, already-trained
leaf detector. Loaded once at controller startup — reloading a YOLO
checkpoint every simulation step (every ~32-64ms) would make the sim
crawl and would defeat the point: the whole credibility argument for this
simulation is that the "brain" is 100% the real leaf_v3 model, unmodified,
just fed frames from a virtual camera instead of a real one.
"""

import os
import sys

import numpy as np

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.inference.predict import Detection, draw_detections  # noqa: E402
from ultralytics import YOLO  # noqa: E402


class LeafVisionBridge:
    """Wraps the real leaf detector for per-frame use inside a Webots
    controller loop. Same Detection dataclass and draw_detections used
    everywhere else in this project — nothing simulation-specific about
    the detection logic itself."""

    def __init__(self, weights_path: str, conf: float = 0.25, class_names=None):
        if not os.path.exists(weights_path):
            raise FileNotFoundError(
                f"leaf weights not found at {weights_path} — copy the real "
                f"leaf_v3 (or leaf_v3_ft) best.pt here, or set LEAF_WEIGHTS_PATH."
            )
        self.model = YOLO(weights_path)
        self.conf = conf
        self.class_names = class_names or ["leaf"]

    def detect(self, frame_bgr: np.ndarray):
        results = self.model.predict(frame_bgr, conf=self.conf, verbose=False)[0]
        return [
            Detection(box=tuple(b.xyxy[0].tolist()), conf=float(b.conf[0]), cls=int(b.cls[0]))
            for b in results.boxes
        ]

    def annotate(self, frame_bgr: np.ndarray, detections) -> np.ndarray:
        return draw_detections(frame_bgr, detections, self.class_names)


def camera_image_to_bgr(camera) -> np.ndarray:
    """Convert a Webots Camera device's raw image buffer (BGRA byte
    order) to a plain BGR numpy array — the format OpenCV/YOLO expect.
    This is the standard Webots Python API conversion; getImage() returns
    raw bytes, not a ready-made array."""
    raw = camera.getImage()
    height, width = camera.getHeight(), camera.getWidth()
    frame_bgra = np.frombuffer(raw, np.uint8).reshape((height, width, 4))
    return frame_bgra[:, :, :3].copy()
