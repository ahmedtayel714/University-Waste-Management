"""Harvest object-free background patches (ground, soil, pavement) from
datasets we already have on disk, instead of sourcing a new "empty ground"
dataset — none exists publicly (verified before building this). Two
sources:

- The crop/weed dataset (Track A): every pixel *outside* the green mask is,
  by construction, soil/background — we just need patches far from any
  labeled crop/weed bounding box too, so we don't accidentally harvest a
  patch that's mostly soil but grazes a plant.
- The TACO waste dataset: patches taken away from every annotated litter
  bounding box are plausible ground/scene background.

Both are real photos of real outdoor ground, just repurposed — a better
domain match for a campus setting than a generic texture dataset would be.
"""

import random
from pathlib import Path

import cv2
import numpy as np

from ..preprocessing.hsv_mask import MaskConfig, green_mask


def _boxes_from_yolo_label(label_path: Path, img_w: int, img_h: int) -> list:
    boxes = []
    if not label_path.exists():
        return boxes
    for line in label_path.read_text().splitlines():
        if not line.strip():
            continue
        _, cx, cy, bw, bh = line.split()[:5]
        cx, cy, bw, bh = float(cx) * img_w, float(cy) * img_h, float(bw) * img_w, float(bh) * img_h
        boxes.append((cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2))
    return boxes


def _patch_overlaps_boxes(x0, y0, x1, y1, boxes) -> bool:
    for bx0, by0, bx1, by1 in boxes:
        if x0 < bx1 and x1 > bx0 and y0 < by1 and y1 > by0:
            return True
    return False


def harvest_from_green_dataset(
    image_label_pairs: list,
    out_dir: Path,
    patch_size: int = 256,
    patches_per_image: int = 2,
    max_green_ratio: float = 0.03,
    mask_config: MaskConfig = None,
    seed: int = 42,
    max_source_images: int = 150,
) -> list:
    """Sample patch_size x patch_size crops from crop/weed images that are
    (a) almost entirely non-green (max_green_ratio ceiling) and (b) don't
    overlap any labeled bounding box, i.e. soil/background only.

    max_source_images caps how many *source* images get opened at all
    (randomly sampled) — this reads full image content (cv2.imread), not
    just filenames, so on a Drive-mounted image_label_pairs list an
    unbounded pass here means hundreds/thousands of network round-trips.
    We only need enough patches for background variety, not one from
    every source image; 150 is already generous for that. Set None for
    the old unbounded behavior (fine if the source images are local)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    written = []

    pairs = image_label_pairs
    if max_source_images is not None and len(pairs) > max_source_images:
        pairs = rng.sample(list(pairs), max_source_images)

    for img_path, lbl_path in pairs:
        image = cv2.imread(str(img_path))
        if image is None:
            continue
        h, w = image.shape[:2]
        if h < patch_size or w < patch_size:
            continue
        boxes = _boxes_from_yolo_label(Path(lbl_path), w, h)

        attempts, found = 0, 0
        while found < patches_per_image and attempts < patches_per_image * 20:
            attempts += 1
            x0 = rng.randint(0, w - patch_size)
            y0 = rng.randint(0, h - patch_size)
            x1, y1 = x0 + patch_size, y0 + patch_size
            if _patch_overlaps_boxes(x0, y0, x1, y1, boxes):
                continue

            patch = image[y0:y1, x0:x1]
            mask = green_mask(patch, mask_config)
            green_ratio = np.count_nonzero(mask) / mask.size
            if green_ratio > max_green_ratio:
                continue

            out_path = out_dir / f"{Path(img_path).stem}_bg{found}.jpg"
            cv2.imwrite(str(out_path), patch)
            written.append(out_path)
            found += 1

    return written


def harvest_from_boxed_dataset(
    image_label_pairs: list,
    out_dir: Path,
    patch_size: int = 256,
    patches_per_image: int = 2,
    seed: int = 42,
    max_source_images: int = 150,
) -> list:
    """Sample patches from a YOLO-labeled dataset (e.g. TACO) that avoid
    every labeled bounding box — generic version for sources without a
    color-based background test. See harvest_from_green_dataset's
    max_source_images docstring — same reasoning, same default."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    written = []

    pairs = image_label_pairs
    if max_source_images is not None and len(pairs) > max_source_images:
        pairs = rng.sample(list(pairs), max_source_images)

    for img_path, lbl_path in pairs:
        image = cv2.imread(str(img_path))
        if image is None:
            continue
        h, w = image.shape[:2]
        if h < patch_size or w < patch_size:
            continue
        boxes = _boxes_from_yolo_label(Path(lbl_path), w, h)

        attempts, found = 0, 0
        while found < patches_per_image and attempts < patches_per_image * 20:
            attempts += 1
            x0 = rng.randint(0, w - patch_size)
            y0 = rng.randint(0, h - patch_size)
            x1, y1 = x0 + patch_size, y0 + patch_size
            if _patch_overlaps_boxes(x0, y0, x1, y1, boxes):
                continue

            patch = image[y0:y1, x0:x1]
            out_path = out_dir / f"{Path(img_path).stem}_bg{found}.jpg"
            cv2.imwrite(str(out_path), patch)
            written.append(out_path)
            found += 1

    return written
