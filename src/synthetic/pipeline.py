"""Single entry point for the whole Track B dataset pipeline: cutout
extraction (leaf + waste) -> background harvesting -> synthetic scene
generation. Wraps cutout_extractor.py, background_harvester.py, and
generate_dataset.py behind one call so building or rebuilding the leaf
dataset is one function call from a notebook or script, not six separate
cells wired together by hand.

The individual modules stay independently usable — this is a convenience
layer on top, not a replacement API."""

import random
from dataclasses import dataclass
from pathlib import Path

from .background_harvester import harvest_from_boxed_dataset, harvest_from_green_dataset
from .cutout_extractor import cutout_from_bbox, extract_cutouts_from_yolo_seg_dataset
from .generate_dataset import DEFAULT_DIFFICULTY_MIX, generate_synthetic_dataset
from ..preprocessing.dataset_prep import discover_pairs
from ..preprocessing.hsv_mask import MaskConfig


@dataclass
class LeafPipelineResult:
    leaf_cutout_paths: list
    waste_cutout_paths: list
    background_paths: list
    dataset_dir: Path
    data_yaml: Path


def build_leaf_dataset(
    leaf_seg_raw_dir: Path,
    veg_dataset_pairs: list,
    waste_dataset_pairs: list,
    work_dir: Path,
    n_train: int = 900,
    n_val: int = 150,
    n_test: int = 150,
    n_leaves_range: tuple = (2, 8),
    canvas_size: tuple = (640, 640),
    difficulty_mix: dict = None,
    mask_config: MaskConfig = None,
    waste_cutout_cap: int = 150,
    veg_background_source_cap: int = 150,
    waste_background_source_cap: int = 150,
    seed: int = 42,
) -> LeafPipelineResult:
    """Run the full Track B dataset pipeline in one call.

    leaf_seg_raw_dir: downloaded YOLO-seg leaf dataset (e.g. from Roboflow).
    veg_dataset_pairs: discover_pairs() output from Track A's crop/weed
        dataset — used only as a source of soil/background patches.
    waste_dataset_pairs: discover_pairs() output from the TACO download —
        used for both waste cutouts (clutter) and background patches.
    work_dir: local staging directory (should be on local disk, not Drive —
        see README's local-disk-staging note; this function doesn't care,
        it just writes wherever it's told).

    veg_dataset_pairs/waste_dataset_pairs typically point at Drive-mounted
    raw folders (they're already-downloaded Track A/waste data, never
    re-staged locally) — every image this function actually *opens*
    (cv2.imread, not just the filename) is one Drive network round-trip.
    waste_cutout_cap/veg_background_source_cap/waste_background_source_cap
    all bound how many source images get opened, via random sampling —
    without these caps this function silently reads every image in both
    datasets (1000s of Drive round-trips, easily 10+ minutes) for what
    only needs a few hundred output images. Lower them further if it's
    still slow; raise them only if the source datasets are also local.
    """
    work_dir = Path(work_dir)
    difficulty_mix = difficulty_mix or DEFAULT_DIFFICULTY_MIX

    leaf_cutout_dir = work_dir / "leaf_cutouts"
    leaf_cutout_paths = extract_cutouts_from_yolo_seg_dataset(Path(leaf_seg_raw_dir), leaf_cutout_dir)
    if not leaf_cutout_paths:
        raise RuntimeError(
            f"No leaf cutouts extracted from {leaf_seg_raw_dir} — check the download "
            "actually contains YOLO-seg polygon labels, not just boxes."
        )

    waste_cutout_dir = work_dir / "waste_cutouts"
    waste_cutout_dir.mkdir(parents=True, exist_ok=True)
    waste_cutout_paths = []
    rng = random.Random(seed)
    sampled_waste_pairs = (
        rng.sample(list(waste_dataset_pairs), waste_cutout_cap)
        if waste_cutout_cap is not None and len(waste_dataset_pairs) > waste_cutout_cap
        else waste_dataset_pairs
    )
    for img_path, lbl_path in sampled_waste_pairs:
        import cv2

        image = cv2.imread(str(img_path))
        if image is None:
            continue
        h, w = image.shape[:2]
        for i, line in enumerate(lbl_path.read_text().splitlines()):
            if not line.strip():
                continue
            _, cx, cy, bw, bh = line.split()[:5]
            cx, cy, bw, bh = float(cx) * w, float(cy) * h, float(bw) * w, float(bh) * h
            box = (cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2)
            try:
                cutout = cutout_from_bbox(image, box)
            except ValueError:
                continue
            out_path = waste_cutout_dir / f"{img_path.stem}_{i}.png"
            cv2.imwrite(str(out_path), cutout)
            waste_cutout_paths.append(out_path)

    background_dir = work_dir / "backgrounds"
    bg_from_green = harvest_from_green_dataset(
        veg_dataset_pairs, background_dir / "from_veg", patch_size=256, patches_per_image=2,
        mask_config=mask_config, max_source_images=veg_background_source_cap,
    )
    bg_from_waste = harvest_from_boxed_dataset(
        waste_dataset_pairs, background_dir / "from_waste", patch_size=256, patches_per_image=1,
        max_source_images=waste_background_source_cap,
    )
    background_paths = bg_from_green + bg_from_waste
    if not background_paths:
        raise RuntimeError("No background patches harvested — check veg_dataset_pairs/waste_dataset_pairs are non-empty.")

    dataset_dir = work_dir / "leaf_synthetic"
    generate_synthetic_dataset(
        leaf_cutout_paths, background_paths, dataset_dir,
        waste_cutout_paths=waste_cutout_paths,
        n_train=n_train, n_val=n_val, n_test=n_test,
        n_leaves_range=n_leaves_range, canvas_size=canvas_size,
        difficulty_mix=difficulty_mix, seed=seed,
    )

    return LeafPipelineResult(
        leaf_cutout_paths=leaf_cutout_paths,
        waste_cutout_paths=waste_cutout_paths,
        background_paths=background_paths,
        dataset_dir=dataset_dir,
        data_yaml=dataset_dir / "data.yaml",
    )
