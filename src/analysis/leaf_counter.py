"""Per-leaf instance splitting and counting via distance-transform watershed
on the HSV green mask. A classical-CV addition, not a trained model — no new
dataset or GPU time required, at the cost of accuracy on heavily overlapping
or thin/wispy leaves (the distance-transform 'neck' between two touching
blobs has to be visible in the mask for the split to happen; leaves that
fully overlap in the 2D projection will under-count)."""

from dataclasses import dataclass

import cv2
import numpy as np

from ..preprocessing.hsv_mask import MaskConfig, green_mask


@dataclass
class LeafCountResult:
    count: int
    label_map: np.ndarray  # same H,W as input; 0 = background, >=2 = leaf id


def count_leaves(
    image_bgr: np.ndarray,
    config: MaskConfig = None,
    min_area: int = 120,
    fg_ratio: float = 0.35,
) -> LeafCountResult:
    """Segment individual leaf blobs from the green mask. fg_ratio controls
    how conservative the 'sure foreground' seed threshold is (as a fraction
    of the max distance-transform value) — raise it if touching leaves are
    being merged into one count, lower it if single leaves are being split
    into several."""
    mask = green_mask(image_bgr, config)

    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    if dist.max() == 0:
        return LeafCountResult(0, np.zeros(mask.shape, dtype=np.int32))

    _, sure_fg = cv2.threshold(dist, fg_ratio * dist.max(), 255, 0)
    sure_fg = sure_fg.astype(np.uint8)

    kernel = np.ones((3, 3), np.uint8)
    sure_bg = cv2.dilate(mask, kernel, iterations=3)
    unknown = cv2.subtract(sure_bg, sure_fg)

    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    cv2.watershed(image_bgr.copy(), markers)

    labels, counts = np.unique(markers, return_counts=True)
    # label 1 is background (from connectedComponents' zero relabeled to 1),
    # label -1 is watershed boundary pixels — only >=2 are candidate leaves.
    leaf_ids = [lbl for lbl, cnt in zip(labels, counts) if lbl >= 2 and cnt >= min_area]

    label_map = np.where(np.isin(markers, leaf_ids), markers, 0).astype(np.int32)
    return LeafCountResult(len(leaf_ids), label_map)


def render_leaf_overlay(image_bgr: np.ndarray, label_map: np.ndarray, alpha: float = 0.55) -> np.ndarray:
    """Tint each leaf id a distinct color, blended over the original image."""
    color_layer = image_bgr.copy()
    rng = np.random.default_rng(42)
    for leaf_id in np.unique(label_map):
        if leaf_id <= 0:
            continue
        color = rng.integers(60, 255, size=3).tolist()
        color_layer[label_map == leaf_id] = color
    return cv2.addWeighted(image_bgr, 1 - alpha, color_layer, alpha, 0)


def count_leaves_in_regions(
    image_bgr: np.ndarray,
    boxes: list,
    config: MaskConfig = None,
    min_area: int = 120,
    fg_ratio: float = 0.35,
) -> tuple[list, np.ndarray, int]:
    """Run count_leaves independently inside each xyxy box (so leaves from
    different detected plants aren't merged), and composite a full-image
    overlay. Falls back to whole-image counting if boxes is empty. Returns
    (per_box_counts, full_image_overlay, total_count)."""
    if not boxes:
        result = count_leaves(image_bgr, config, min_area, fg_ratio)
        overlay = render_leaf_overlay(image_bgr, result.label_map)
        return [result.count], overlay, result.count

    overlay = image_bgr.copy()
    per_box_counts = []
    total = 0
    for box in boxes:
        x1, y1, x2, y2 = [int(v) for v in box]
        x1, y1 = max(x1, 0), max(y1, 0)
        x2, y2 = min(x2, image_bgr.shape[1]), min(y2, image_bgr.shape[0])
        crop = image_bgr[y1:y2, x1:x2]
        if crop.size == 0:
            per_box_counts.append(0)
            continue
        result = count_leaves(crop, config, min_area, fg_ratio)
        overlay[y1:y2, x1:x2] = render_leaf_overlay(crop, result.label_map)
        per_box_counts.append(result.count)
        total += result.count

    return per_box_counts, overlay, total
