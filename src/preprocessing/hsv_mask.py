"""HSV-based green color masking for isolating vegetation from soil/dry-straw background."""

from dataclasses import dataclass

import cv2
import numpy as np

# OpenCV HSV ranges: H in [0,179], S/V in [0,255].
# Tuned to admit yellow-green through blue-green vegetation while excluding
# dry straw (~H 20-30, high V, low-mid S) and soil (low S, brown hue).
DEFAULT_LOWER_GREEN = np.array([28, 35, 30], dtype=np.uint8)
DEFAULT_UPPER_GREEN = np.array([95, 255, 255], dtype=np.uint8)


@dataclass
class MaskConfig:
    lower: np.ndarray = None
    upper: np.ndarray = None
    open_kernel: int = 3
    open_iterations: int = 1
    close_kernel: int = 7
    close_iterations: int = 2
    blur_kernel: int = 5

    def __post_init__(self):
        if self.lower is None:
            self.lower = DEFAULT_LOWER_GREEN
        if self.upper is None:
            self.upper = DEFAULT_UPPER_GREEN


def green_mask(image_bgr: np.ndarray, config: MaskConfig = None) -> np.ndarray:
    """Return a binary (0/255) mask of green-vegetation pixels in image_bgr."""
    config = config or MaskConfig()

    blurred = cv2.GaussianBlur(image_bgr, (config.blur_kernel, config.blur_kernel), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, config.lower, config.upper)

    # Opening first removes salt-noise specks (e.g. sky glare) without eroding
    # plant silhouettes; closing after fills small holes inside leaf clusters.
    open_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (config.open_kernel, config.open_kernel))
    close_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (config.close_kernel, config.close_kernel))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_k, iterations=config.open_iterations)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_k, iterations=config.close_iterations)

    return mask


def apply_mask_hard(image_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Zero out every non-vegetation pixel. Maximum noise removal, but the
    resulting hard-edged silhouettes can become a spurious shortcut cue the
    detector learns instead of real vegetation texture — validate with the
    soft variant before trusting mAP gains from this mode alone."""
    return cv2.bitwise_and(image_bgr, image_bgr, mask=mask)


def apply_mask_soft(image_bgr: np.ndarray, mask: np.ndarray, background_scale: float = 0.25) -> np.ndarray:
    """Dim and desaturate background instead of zeroing it, keeping spatial
    context (field horizon, row structure) while still suppressing soil/straw
    color signal. Recommended default for training — softer domain shift from
    the unmasked inference-time images than the hard variant."""
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 1] = np.where(mask > 0, hsv[..., 1], hsv[..., 1] * background_scale)
    hsv[..., 2] = np.where(mask > 0, hsv[..., 2], hsv[..., 2] * (background_scale + 0.35))
    hsv = np.clip(hsv, 0, 255).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def green_pixel_ratio(image_bgr: np.ndarray, box_xyxy, config: MaskConfig = None) -> float:
    """Fraction of green pixels inside a bounding box crop. Used post-inference
    as a domain-specific false-positive filter (see src/inference/predict.py)."""
    x1, y1, x2, y2 = [int(v) for v in box_xyxy]
    x1, y1 = max(x1, 0), max(y1, 0)
    x2, y2 = min(x2, image_bgr.shape[1]), min(y2, image_bgr.shape[0])
    crop = image_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return 0.0
    mask = green_mask(crop, config)
    return float(np.count_nonzero(mask)) / mask.size


def visualize_pipeline(image_bgr: np.ndarray, config: MaskConfig = None):
    """Return (mask, hard_masked, soft_masked) for side-by-side inspection."""
    mask = green_mask(image_bgr, config)
    return mask, apply_mask_hard(image_bgr, mask), apply_mask_soft(image_bgr, mask)
