"""Extract RGBA cutouts (transparent background, alpha = object silhouette)
from source images, for use as paste material in the synthetic compositor.

Two paths:
- From an explicit polygon mask (YOLO-seg label format: `class x1 y1 x2 y2
  ...` normalized points) — the accurate path, used when the source dataset
  ships segmentation annotations (e.g. a Roboflow leaf-segmentation export).
- Classical auto-segmentation for plain/simple-background source images that
  have no annotation at all (flood-fill from the corners to find the
  background, refined with GrabCut) — the fallback path, e.g. for raw phone
  photos of a leaf placed on a plain sheet, dropped in later without needing
  new annotation code.
"""

from pathlib import Path

import cv2
import numpy as np


def parse_yolo_seg_line(line: str, img_w: int, img_h: int) -> tuple[int, np.ndarray]:
    """Parse one YOLO-seg label line into (class_id, polygon points in pixel
    coords, shape (N, 2))."""
    parts = line.split()
    cls_id = int(parts[0])
    coords = [float(v) for v in parts[1:]]
    points = np.array(coords, dtype=np.float32).reshape(-1, 2)
    points[:, 0] *= img_w
    points[:, 1] *= img_h
    return cls_id, points.astype(np.int32)


def cutout_from_polygon(image_bgr: np.ndarray, polygon: np.ndarray, pad: int = 4) -> np.ndarray:
    """Return a tightly-cropped RGBA cutout for one polygon instance."""
    h, w = image_bgr.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [polygon], 255)

    x, y, bw, bh = cv2.boundingRect(polygon)
    x0, y0 = max(x - pad, 0), max(y - pad, 0)
    x1, y1 = min(x + bw + pad, w), min(y + bh + pad, h)

    crop = image_bgr[y0:y1, x0:x1]
    mask_crop = mask[y0:y1, x0:x1]

    rgba = cv2.cvtColor(crop, cv2.COLOR_BGR2BGRA)
    rgba[:, :, 3] = mask_crop
    return rgba


def extract_cutouts_from_yolo_seg_dataset(dataset_dir: Path, out_dir: Path, class_filter: int = None) -> list:
    """Walk a YOLO-seg formatted dataset (images/ + matching polygon .txt
    labels) and write one RGBA cutout PNG per labeled instance. class_filter
    restricts to one source class id if the export has more than one."""
    dataset_dir, out_dir = Path(dataset_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    image_exts = {".jpg", ".jpeg", ".png", ".bmp"}
    images = {p.stem: p for p in dataset_dir.rglob("*") if p.suffix.lower() in image_exts}
    labels = {p.stem: p for p in dataset_dir.rglob("*.txt") if p.stem in images}

    written = []
    for stem, label_path in labels.items():
        image = cv2.imread(str(images[stem]))
        if image is None:
            continue
        h, w = image.shape[:2]

        for i, line in enumerate(label_path.read_text().splitlines()):
            if not line.strip():
                continue
            cls_id, polygon = parse_yolo_seg_line(line, w, h)
            if class_filter is not None and cls_id != class_filter:
                continue
            if len(polygon) < 3:
                continue
            cutout = cutout_from_polygon(image, polygon)
            if cutout.shape[0] < 8 or cutout.shape[1] < 8:
                continue
            out_path = out_dir / f"{stem}_{i}.png"
            cv2.imwrite(str(out_path), cutout)
            written.append(out_path)

    return written


def cutout_from_bbox(image_bgr: np.ndarray, box_xyxy, feather: int = 3) -> np.ndarray:
    """Rectangular RGBA cutout for a bounding-box source (no mask available —
    e.g. TACO waste items). Edges are alpha-feathered so the paste doesn't
    leave a hard rectangular seam."""
    x1, y1, x2, y2 = [int(v) for v in box_xyxy]
    x1, y1 = max(x1, 0), max(y1, 0)
    x2, y2 = min(x2, image_bgr.shape[1]), min(y2, image_bgr.shape[0])
    crop = image_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        raise ValueError("empty crop region")

    h, w = crop.shape[:2]
    mask = np.full((h, w), 255, dtype=np.uint8)
    if feather > 0 and min(h, w) > feather * 2:
        mask = cv2.rectangle(np.zeros((h, w), dtype=np.uint8), (feather, feather), (w - feather, h - feather), 255, -1)
        mask = cv2.GaussianBlur(mask, (feather * 2 + 1, feather * 2 + 1), 0)

    rgba = cv2.cvtColor(crop, cv2.COLOR_BGR2BGRA)
    rgba[:, :, 3] = mask
    return rgba


def auto_segment_plain_background(image_bgr: np.ndarray, tolerance: int = 18, grabcut_iters: int = 3) -> np.ndarray:
    """Fallback foreground extraction for a single-object image on a roughly
    uniform background (e.g. a leaf photographed on a plain sheet). Seeds
    GrabCut with a flood-fill-from-corners estimate of the background, since
    plain-background catalog-style photos are a much better fit for that
    than a generic saliency guess. Returns a binary mask (0/255)."""
    h, w = image_bgr.shape[:2]
    flood_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
    bg_estimate = np.zeros((h, w), dtype=np.uint8)

    corners = [(0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)]
    work = image_bgr.copy()
    for cx, cy in corners:
        flood_mask[:] = 0
        cv2.floodFill(
            work, flood_mask, (cx, cy), 255,
            loDiff=(tolerance,) * 3, upDiff=(tolerance,) * 3,
            flags=cv2.FLOODFILL_FIXED_RANGE,
        )
    bg_estimate = flood_mask[1:-1, 1:-1] * 255

    gc_mask = np.where(bg_estimate > 0, cv2.GC_PR_BGD, cv2.GC_PR_FGD).astype("uint8")
    bgd_model = np.zeros((1, 65), dtype=np.float64)
    fgd_model = np.zeros((1, 65), dtype=np.float64)
    try:
        cv2.grabCut(image_bgr, gc_mask, None, bgd_model, fgd_model, grabcut_iters, cv2.GC_INIT_WITH_MASK)
    except cv2.error:
        return 255 - bg_estimate

    return np.where((gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
