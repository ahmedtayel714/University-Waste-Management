"""Orchestrate the full synthetic dataset build: sample backgrounds, paste
leaf (+ waste-clutter) cutouts via the compositor, and write a standard YOLO
train/val/test tree with a single 'leaf' class — the Stage 1 deliverable
from the robotic-collection system spec's dataset strategy."""

import random
from pathlib import Path

import cv2
from tqdm import tqdm

from .compositor import compose_scene
from ..preprocessing.dataset_prep import write_data_yaml

# 'scatter' gets the largest share: real reference photos of campus paths
# and sidewalks predominantly show individually scattered leaves (not
# mounded piles), so that pattern should dominate what the model trains on.
DEFAULT_DIFFICULTY_MIX = {"easy": 0.15, "medium": 0.2, "hard": 0.15, "pile": 0.15, "scatter": 0.35}


def _load_rgba(paths: list) -> list:
    cutouts = []
    for p in paths:
        img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        if img is None or img.shape[2] != 4:
            continue
        cutouts.append(img)
    return cutouts


def _sample_difficulty(rng: random.Random, mix: dict) -> str:
    names, weights = zip(*mix.items())
    return rng.choices(names, weights=weights, k=1)[0]


def _fit_background(image, canvas_size: tuple, rng: random.Random):
    h, w = image.shape[:2]
    cw, ch = canvas_size
    if h < ch or w < cw:
        scale = max(ch / h, cw / w) * 1.05
        image = cv2.resize(image, (int(w * scale) + 1, int(h * scale) + 1))
        h, w = image.shape[:2]
    x0 = rng.randint(0, w - cw)
    y0 = rng.randint(0, h - ch)
    return image[y0:y0 + ch, x0:x0 + cw]


def _generate_split(
    split_name: str,
    n_images: int,
    leaf_cutouts: list,
    waste_cutouts: list,
    background_paths: list,
    out_dir: Path,
    difficulty_mix: dict,
    n_leaves_range: tuple,
    canvas_size: tuple,
    rng: random.Random,
):
    img_dir = out_dir / "images" / split_name
    lbl_dir = out_dir / "labels" / split_name
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    for i in tqdm(range(n_images), desc=f"synth:{split_name}"):
        bg_path = rng.choice(background_paths)
        bg = cv2.imread(str(bg_path))
        if bg is None:
            continue
        bg = _fit_background(bg, canvas_size, rng)

        difficulty = _sample_difficulty(rng, difficulty_mix)
        result = compose_scene(
            bg, leaf_cutouts, waste_cutouts,
            n_leaves=n_leaves_range, difficulty=difficulty, seed=rng.randint(0, 2**31),
        )

        name = f"{split_name}_{i:05d}_{difficulty}"
        cv2.imwrite(str(img_dir / f"{name}.jpg"), result.image)
        lines = [f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}" for cx, cy, bw, bh in result.boxes]
        (lbl_dir / f"{name}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))


def generate_synthetic_dataset(
    leaf_cutout_paths: list,
    background_paths: list,
    out_dir: Path,
    waste_cutout_paths: list = None,
    n_train: int = 800,
    n_val: int = 150,
    n_test: int = 150,
    difficulty_mix: dict = None,
    n_leaves_range: tuple = None,
    canvas_size: tuple = (640, 640),
    seed: int = 42,
) -> Path:
    """n_leaves_range=None (default) lets each difficulty tier use its own
    density (see compositor.DIFFICULTY_PRESETS) — 'pile' scenes get far
    more leaves than 'easy' ones by design. Pass an explicit tuple to
    override every tier uniformly, matching the original single-density
    behavior from before the 'pile' tier existed."""
    if not leaf_cutout_paths:
        raise ValueError("leaf_cutout_paths is empty — build the cutout library first")
    if not background_paths:
        raise ValueError("background_paths is empty — harvest backgrounds first")

    difficulty_mix = difficulty_mix or DEFAULT_DIFFICULTY_MIX
    out_dir = Path(out_dir)
    rng = random.Random(seed)

    leaf_cutouts = _load_rgba(leaf_cutout_paths)
    waste_cutouts = _load_rgba(waste_cutout_paths) if waste_cutout_paths else []
    if not leaf_cutouts:
        raise ValueError("no valid RGBA leaf cutouts loaded")

    for split_name, n in [("train", n_train), ("val", n_val), ("test", n_test)]:
        _generate_split(
            split_name, n, leaf_cutouts, waste_cutouts, background_paths,
            out_dir, difficulty_mix, n_leaves_range, canvas_size, rng,
        )

    write_data_yaml(out_dir / "data.yaml", out_dir, names=["leaf"])
    return out_dir
