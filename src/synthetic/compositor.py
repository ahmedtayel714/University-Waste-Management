"""Controlled synthetic scene generation: paste real leaf cutouts (the
labeled target) and real waste cutouts (unlabeled visual clutter — per the
spec, waste is not a detection class here, just realistic distraction) onto
a real harvested background, with randomized geometry so ground-truth boxes
come out of the paste operation for free — no manual annotation.

Waste cutouts are never labeled: the spec is explicit that the only
detection class is "leaf"; plastic/paper/etc. exist in scenes purely to
make the detector robust to visual clutter, matching Section 5's
requirement to detect "leaves mixed with plastic" / "leaves mixed with
paper" without ever asking the model to name those materials.
"""

import random
from dataclasses import dataclass

import cv2
import numpy as np

DIFFICULTY_PRESETS = {
    # n_leaves is this tier's own density range, used when compose_scene's
    # n_leaves argument is left as None (see compose_scene) — an explicit
    # n_leaves still overrides every tier uniformly, for backward
    # compatibility with earlier datasets/results built that way.
    # cluster_prob: chance this scene concentrates leaves around 1-2 mound
    # centers (Gaussian scatter, see _sample_position) instead of placing
    # each leaf fully independently across the whole canvas. Without this,
    # even a high leaf count just looks like thin confetti spread over the
    # entire frame — real piles are spatially concentrated, not just
    # numerous. cluster_spread_frac is the Gaussian std as a fraction of
    # canvas size (smaller = tighter mound).
    "easy": dict(scale_range=(0.6, 1.0), rotation_range=(-20, 20), overlap_prob=0.05,
                 edge_crop_prob=0.0, blur_prob=0.1, shadow_prob=0.3, waste_count=(0, 1),
                 n_leaves=(2, 6), dry_leaf_prob=0.3, cluster_prob=0.0,
                 n_clusters=(1, 1), cluster_spread_frac=0.2),
    "medium": dict(scale_range=(0.35, 1.1), rotation_range=(-45, 45), overlap_prob=0.25,
                   edge_crop_prob=0.1, blur_prob=0.25, shadow_prob=0.5, waste_count=(0, 3),
                   n_leaves=(6, 14), dry_leaf_prob=0.4, cluster_prob=0.2,
                   n_clusters=(1, 2), cluster_spread_frac=0.18),
    "hard": dict(scale_range=(0.25, 1.2), rotation_range=(-90, 90), overlap_prob=0.5,
                 edge_crop_prob=0.25, blur_prob=0.4, shadow_prob=0.6, waste_count=(1, 5),
                 n_leaves=(12, 22), dry_leaf_prob=0.5, cluster_prob=0.45,
                 n_clusters=(1, 2), cluster_spread_frac=0.15),
    # Dense leaf-litter / pile scenes — real photos of leaf piles, garden-
    # waste bags, and litter holes look nothing like a handful of scattered
    # leaves; this tier exists specifically to cover that case. dry_leaf_prob
    # is high since real piles skew heavily toward dry/brown leaves, and
    # cluster_prob is near-1 so the leaf count actually reads as a mound.
    "pile": dict(scale_range=(0.3, 0.9), rotation_range=(-180, 180), overlap_prob=0.85,
                 edge_crop_prob=0.35, blur_prob=0.3, shadow_prob=0.4, waste_count=(0, 4),
                 n_leaves=(20, 45), dry_leaf_prob=0.7, cluster_prob=0.9,
                 n_clusters=(1, 2), cluster_spread_frac=0.13),
}


@dataclass
class SceneResult:
    image: np.ndarray
    boxes: list  # list of (cx, cy, w, h) normalized YOLO coords, class "leaf" implicit


def _tint_dry(cutout_rgba: np.ndarray, rng: random.Random) -> np.ndarray:
    """Shift a leaf cutout's color toward dry brown/yellow, roughly
    simulating a dead/fallen leaf from a fresh-green source cutout —
    the leaf-segmentation source dataset skews toward fresh green leaves,
    so this is how color diversity gets into the synthetic set without a
    second dataset download. Alpha channel is untouched."""
    bgr = cutout_rgba[:, :, :3]
    alpha = cutout_rgba[:, :, 3]

    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    # Target hue ~15-30 (yellow/orange/brown in OpenCV's 0-179 scale),
    # desaturate and darken somewhat — a green leaf (hue ~35-85) pulled
    # this direction reads as dried/dead rather than a different plant.
    target_hue = rng.uniform(12, 28)
    pull = rng.uniform(0.55, 0.9)
    hsv[:, :, 0] = hsv[:, :, 0] * (1 - pull) + target_hue * pull
    hsv[:, :, 1] *= rng.uniform(0.45, 0.75)
    hsv[:, :, 2] *= rng.uniform(0.7, 0.95)
    hsv = np.clip(hsv, 0, 255).astype(np.uint8)

    tinted_bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return np.dstack([tinted_bgr, alpha])


def _transform_cutout(cutout_rgba: np.ndarray, scale: float, angle: float) -> np.ndarray:
    h, w = cutout_rgba.shape[:2]
    new_w, new_h = max(int(w * scale), 4), max(int(h * scale), 4)
    resized = cv2.resize(cutout_rgba, (new_w, new_h), interpolation=cv2.INTER_AREA)

    diag = int(np.ceil((new_w ** 2 + new_h ** 2) ** 0.5))
    canvas = np.zeros((diag, diag, 4), dtype=np.uint8)
    ox, oy = (diag - new_w) // 2, (diag - new_h) // 2
    canvas[oy:oy + new_h, ox:ox + new_w] = resized

    center = (diag / 2, diag / 2)
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(canvas, rot_mat, (diag, diag), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0, 0))

    alpha = rotated[:, :, 3]
    alpha = cv2.GaussianBlur(alpha, (3, 3), 0)
    rotated[:, :, 3] = alpha

    ys, xs = np.where(alpha > 8)
    if len(xs) == 0:
        return rotated
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    return rotated[y0:y1, x0:x1]


def _paste_with_shadow(background: np.ndarray, cutout_rgba: np.ndarray, x: int, y: int, with_shadow: bool):
    h, w = background.shape[:2]
    ch, cw = cutout_rgba.shape[:2]

    if with_shadow:
        shadow_offset = max(2, int(min(ch, cw) * 0.05))
        sx, sy = x + shadow_offset, y + shadow_offset
        _alpha_blend(background, np.zeros((ch, cw, 3), dtype=np.uint8),
                     (cutout_rgba[:, :, 3].astype(np.float32) * 0.35).astype(np.uint8), sx, sy)

    alpha = cutout_rgba[:, :, 3]
    _alpha_blend(background, cutout_rgba[:, :, :3], alpha, x, y)


def _alpha_blend(background: np.ndarray, fg_bgr: np.ndarray, alpha: np.ndarray, x: int, y: int):
    h, w = background.shape[:2]
    ch, cw = fg_bgr.shape[:2]

    bx0, by0 = max(x, 0), max(y, 0)
    bx1, by1 = min(x + cw, w), min(y + ch, h)
    if bx1 <= bx0 or by1 <= by0:
        return

    fx0, fy0 = bx0 - x, by0 - y
    fx1, fy1 = fx0 + (bx1 - bx0), fy0 + (by1 - by0)

    roi = background[by0:by1, bx0:bx1].astype(np.float32)
    fg = fg_bgr[fy0:fy1, fx0:fx1].astype(np.float32)
    a = (alpha[fy0:fy1, fx0:fx1].astype(np.float32) / 255.0)[..., None]

    background[by0:by1, bx0:bx1] = (fg * a + roi * (1 - a)).astype(np.uint8)


def _sample_position(rng: random.Random, w: int, h: int, cw: int, ch: int,
                      cluster_centers: list, spread: float, margin_x: int, margin_y: int) -> tuple:
    """Pick a paste position for a cutout of size (cw, ch) on a w x h
    canvas. With cluster_centers, samples from a Gaussian around a random
    chosen center (mound effect); otherwise uniform across the canvas
    (the original scattered behavior)."""
    if cluster_centers:
        ccx, ccy = rng.choice(cluster_centers)
        x = int(rng.gauss(ccx, spread) - cw / 2)
        y = int(rng.gauss(ccy, spread) - ch / 2)
        x = max(-margin_x, min(x, w - cw + margin_x))
        y = max(-margin_y, min(y, h - ch + margin_y))
        return x, y
    return (
        rng.randint(-margin_x, max(w - cw + margin_x, -margin_x)),
        rng.randint(-margin_y, max(h - ch + margin_y, -margin_y)),
    )


def compose_scene(
    background_bgr: np.ndarray,
    leaf_cutouts: list,
    waste_cutouts: list = None,
    n_leaves: tuple = None,
    difficulty: str = "medium",
    seed: int = None,
) -> SceneResult:
    """Paste a random number of leaf cutouts (labeled) and difficulty-scaled
    waste cutouts (unlabeled clutter) onto background_bgr. Returns the
    composited image and YOLO-normalized boxes for the leaves only.

    n_leaves=None (default) uses the difficulty preset's own density range
    — 'pile' is dramatically denser than 'easy' by design, simulating real
    leaf-litter photos rather than a handful of scattered leaves. Pass an
    explicit n_leaves to override every tier uniformly, matching the
    original single-density behavior."""
    if difficulty not in DIFFICULTY_PRESETS:
        raise ValueError(f"difficulty must be one of {list(DIFFICULTY_PRESETS)}")
    preset = DIFFICULTY_PRESETS[difficulty]
    rng = random.Random(seed)
    leaves_range = n_leaves if n_leaves is not None else preset["n_leaves"]

    canvas = background_bgr.copy()
    h, w = canvas.shape[:2]
    boxes = []

    cluster_centers = []
    if rng.random() < preset["cluster_prob"]:
        n_clusters = rng.randint(*preset["n_clusters"])
        cluster_centers = [
            (rng.uniform(0.2 * w, 0.8 * w), rng.uniform(0.2 * h, 0.8 * h))
            for _ in range(n_clusters)
        ]
    spread = preset["cluster_spread_frac"] * min(w, h)

    n_leaf = rng.randint(*leaves_range)
    for _ in range(n_leaf):
        cutout = rng.choice(leaf_cutouts)
        if rng.random() < preset["dry_leaf_prob"]:
            cutout = _tint_dry(cutout, rng)
        scale = rng.uniform(*preset["scale_range"])
        angle = rng.uniform(*preset["rotation_range"])
        transformed = _transform_cutout(cutout, scale, angle)
        ch, cw = transformed.shape[:2]
        if ch < 4 or cw < 4:
            continue

        edge_crop = rng.random() < preset["edge_crop_prob"]
        margin_x = int(cw * 0.4) if edge_crop else 0
        margin_y = int(ch * 0.4) if edge_crop else 0
        x, y = _sample_position(rng, w, h, cw, ch, cluster_centers, spread, margin_x, margin_y)

        with_shadow = rng.random() < preset["shadow_prob"]
        _paste_with_shadow(canvas, transformed, x, y, with_shadow)

        vis_x0, vis_y0 = max(x, 0), max(y, 0)
        vis_x1, vis_y1 = min(x + cw, w), min(y + ch, h)
        if vis_x1 <= vis_x0 or vis_y1 <= vis_y0:
            continue

        cx = (vis_x0 + vis_x1) / 2 / w
        cy = (vis_y0 + vis_y1) / 2 / h
        bw = (vis_x1 - vis_x0) / w
        bh = (vis_y1 - vis_y0) / h
        boxes.append((cx, cy, bw, bh))

    if waste_cutouts:
        n_waste = rng.randint(*preset["waste_count"])
        for _ in range(n_waste):
            cutout = rng.choice(waste_cutouts)
            scale = rng.uniform(0.5, 1.3)
            angle = rng.uniform(-30, 30)
            transformed = _transform_cutout(cutout, scale, angle)
            ch, cw = transformed.shape[:2]
            if ch < 4 or cw < 4:
                continue
            x = rng.randint(-cw // 3, max(w - cw + cw // 3, -cw // 3))
            y = rng.randint(-ch // 3, max(h - ch + ch // 3, -ch // 3))
            _paste_with_shadow(canvas, transformed, x, y, rng.random() < 0.4)

    if rng.random() < preset["blur_prob"]:
        k = rng.choice([3, 5])
        canvas = cv2.GaussianBlur(canvas, (k, k), 0)

    return SceneResult(image=canvas, boxes=boxes)
