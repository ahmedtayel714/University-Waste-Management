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
    "easy": dict(scale_range=(0.6, 1.0), rotation_range=(-20, 20), overlap_prob=0.05,
                 edge_crop_prob=0.0, blur_prob=0.1, shadow_prob=0.3, waste_count=(0, 1)),
    "medium": dict(scale_range=(0.35, 1.1), rotation_range=(-45, 45), overlap_prob=0.25,
                   edge_crop_prob=0.1, blur_prob=0.25, shadow_prob=0.5, waste_count=(0, 3)),
    "hard": dict(scale_range=(0.2, 1.2), rotation_range=(-90, 90), overlap_prob=0.5,
                 edge_crop_prob=0.25, blur_prob=0.4, shadow_prob=0.6, waste_count=(1, 5)),
}


@dataclass
class SceneResult:
    image: np.ndarray
    boxes: list  # list of (cx, cy, w, h) normalized YOLO coords, class "leaf" implicit


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


def compose_scene(
    background_bgr: np.ndarray,
    leaf_cutouts: list,
    waste_cutouts: list = None,
    n_leaves: tuple = (2, 8),
    difficulty: str = "medium",
    seed: int = None,
) -> SceneResult:
    """Paste a random number of leaf cutouts (labeled) and difficulty-scaled
    waste cutouts (unlabeled clutter) onto background_bgr. Returns the
    composited image and YOLO-normalized boxes for the leaves only."""
    if difficulty not in DIFFICULTY_PRESETS:
        raise ValueError(f"difficulty must be one of {list(DIFFICULTY_PRESETS)}")
    preset = DIFFICULTY_PRESETS[difficulty]
    rng = random.Random(seed)

    canvas = background_bgr.copy()
    h, w = canvas.shape[:2]
    boxes = []

    n_leaf = rng.randint(*n_leaves)
    for _ in range(n_leaf):
        cutout = rng.choice(leaf_cutouts)
        scale = rng.uniform(*preset["scale_range"])
        angle = rng.uniform(*preset["rotation_range"])
        transformed = _transform_cutout(cutout, scale, angle)
        ch, cw = transformed.shape[:2]
        if ch < 4 or cw < 4:
            continue

        edge_crop = rng.random() < preset["edge_crop_prob"]
        margin_x = int(cw * 0.4) if edge_crop else 0
        margin_y = int(ch * 0.4) if edge_crop else 0
        x = rng.randint(-margin_x, max(w - cw + margin_x, -margin_x))
        y = rng.randint(-margin_y, max(h - ch + margin_y, -margin_y))

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
