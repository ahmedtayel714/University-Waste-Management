"""Dataset assembly: discover image/label pairs, merge to a single vegetation
class, split train/val/test, and generate a masked-image variant for the
baseline-vs-masked ablation.

Built for the Kaggle "Crop and Weed Detection Data with Bounding Boxes"
dataset (ravirajsinh45), whose exact on-disk layout varies by download path
(flat co-located images+labels, or separate images/ and labels/ dirs) and
whose class-id convention (commonly 0=crop, 1=weed) should be confirmed with
`inspect_class_distribution` before merging — don't trust it blindly.
"""

import random
import shutil
from collections import Counter
from pathlib import Path

import cv2
import yaml
from tqdm import tqdm

from .hsv_mask import MaskConfig, apply_mask_hard, apply_mask_soft, green_mask

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def discover_pairs(root: Path) -> list[tuple[Path, Path]]:
    """Find (image, label) pairs anywhere under root, matched by filename stem."""
    root = Path(root)
    images = {p.stem: p for p in root.rglob("*") if p.suffix.lower() in IMAGE_EXTS}
    labels = {p.stem: p for p in root.rglob("*.txt") if p.stem in images}
    pairs = [(images[stem], labels[stem]) for stem in labels]
    if not pairs:
        raise FileNotFoundError(f"No matching image/label pairs found under {root}")
    return pairs


def inspect_class_distribution(pairs: list[tuple[Path, Path]]) -> Counter:
    """Count class-id occurrences across all label files — run this first and
    eyeball it before assuming the 0=crop/1=weed convention holds."""
    counts = Counter()
    for _, label_path in pairs:
        for line in label_path.read_text().splitlines():
            if line.strip():
                counts[int(line.split()[0])] += 1
    return counts


def split_dataset(
    pairs: list[tuple[Path, Path]],
    out_dir: Path,
    train: float = 0.7,
    val: float = 0.2,
    test: float = 0.1,
    seed: int = 42,
    merge_to_single_class: bool = True,
) -> Path:
    """Copy images + rewritten labels into a standard YOLO train/val/test tree.
    All class ids in the source labels are collapsed to class 0
    ('green_vegetation') when merge_to_single_class is True."""
    assert abs(train + val + test - 1.0) < 1e-6
    out_dir = Path(out_dir)
    rng = random.Random(seed)
    shuffled = pairs[:]
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_train = int(n * train)
    n_val = int(n * val)
    splits = {
        "train": shuffled[:n_train],
        "val": shuffled[n_train:n_train + n_val],
        "test": shuffled[n_train + n_val:],
    }

    for split_name, split_pairs in splits.items():
        img_dir = out_dir / "images" / split_name
        lbl_dir = out_dir / "labels" / split_name
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        for img_path, lbl_path in tqdm(split_pairs, desc=f"split:{split_name}"):
            shutil.copy2(img_path, img_dir / img_path.name)
            lines = lbl_path.read_text().splitlines()
            if merge_to_single_class:
                lines = [
                    " ".join(["0"] + line.split()[1:])
                    for line in lines
                    if line.strip()
                ]
            (lbl_dir / lbl_path.with_suffix(".txt").name).write_text("\n".join(lines) + "\n")

    return out_dir


def build_masked_variant(
    split_dir: Path,
    out_dir: Path,
    mode: str = "soft",
    config: MaskConfig = None,
) -> Path:
    """Apply HSV green-masking to every image in an already-split YOLO tree
    and mirror the (unchanged) label files. Bounding boxes are unaffected
    since masking never moves pixels, only suppresses background color."""
    if mode not in {"soft", "hard"}:
        raise ValueError("mode must be 'soft' or 'hard'")
    apply_fn = apply_mask_soft if mode == "soft" else apply_mask_hard

    split_dir, out_dir = Path(split_dir), Path(out_dir)
    for split_name in ("train", "val", "test"):
        src_img_dir = split_dir / "images" / split_name
        src_lbl_dir = split_dir / "labels" / split_name
        if not src_img_dir.exists():
            continue

        dst_img_dir = out_dir / "images" / split_name
        dst_lbl_dir = out_dir / "labels" / split_name
        dst_img_dir.mkdir(parents=True, exist_ok=True)
        dst_lbl_dir.mkdir(parents=True, exist_ok=True)

        for img_path in tqdm(list(src_img_dir.iterdir()), desc=f"mask:{split_name}"):
            image = cv2.imread(str(img_path))
            if image is None:
                continue
            mask = green_mask(image, config)
            masked = apply_fn(image, mask)
            cv2.imwrite(str(dst_img_dir / img_path.name), masked)

            lbl_src = src_lbl_dir / img_path.with_suffix(".txt").name
            if lbl_src.exists():
                shutil.copy2(lbl_src, dst_lbl_dir / lbl_src.name)

    return out_dir


def write_data_yaml(path: Path, dataset_root: Path, names: list[str] = None) -> Path:
    names = names or ["green_vegetation"]
    data = {
        "path": str(Path(dataset_root).resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": len(names),
        "names": names,
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False))
    return path
