"""YOLOv8 training wrapper. Runs a single named experiment (e.g. baseline or
masked) with consistent hyperparameters so runs are directly comparable."""

from dataclasses import dataclass, field
from pathlib import Path

from ultralytics import YOLO


@dataclass
class TrainConfig:
    data_yaml: str
    model: str = "yolov8s.pt"
    epochs: int = 100
    imgsz: int = 512
    batch: int = 16
    patience: int = 25
    seed: int = 42
    project: str = "runs"
    name: str = "experiment"
    device: int | str = 0
    # Augmentation — HSV jitter is deliberately reduced vs. Ultralytics
    # defaults because our preprocessing pipeline already encodes a strong
    # color prior (green vegetation); aggressive hue/sat jitter would fight
    # that signal during training on the masked variant.
    hsv_h: float = 0.010
    hsv_s: float = 0.5
    hsv_v: float = 0.3
    degrees: float = 5.0
    translate: float = 0.1
    scale: float = 0.4
    fliplr: float = 0.5
    mosaic: float = 1.0
    extra_overrides: dict = field(default_factory=dict)


def train(config: TrainConfig):
    model = YOLO(config.model)
    overrides = dict(
        data=config.data_yaml,
        epochs=config.epochs,
        imgsz=config.imgsz,
        batch=config.batch,
        patience=config.patience,
        seed=config.seed,
        project=config.project,
        name=config.name,
        device=config.device,
        hsv_h=config.hsv_h,
        hsv_s=config.hsv_s,
        hsv_v=config.hsv_v,
        degrees=config.degrees,
        translate=config.translate,
        scale=config.scale,
        fliplr=config.fliplr,
        mosaic=config.mosaic,
        plots=True,
    )
    overrides.update(config.extra_overrides)
    results = model.train(**overrides)
    return model, results


def validate(weights_path: str, data_yaml: str, imgsz: int = 512):
    model = YOLO(weights_path)
    return model.val(data=data_yaml, imgsz=imgsz, plots=True)


def run_ablation(baseline_yaml: str, masked_yaml: str, base_config: TrainConfig = None):
    """Train the same architecture/hyperparameters on unmasked (baseline) and
    masked datasets back-to-back, so the only variable is the preprocessing."""
    base_config = base_config or TrainConfig(data_yaml=baseline_yaml)

    baseline_cfg = TrainConfig(**{**base_config.__dict__, "data_yaml": baseline_yaml, "name": "baseline"})
    masked_cfg = TrainConfig(**{**base_config.__dict__, "data_yaml": masked_yaml, "name": "masked"})

    _, baseline_results = train(baseline_cfg)
    _, masked_results = train(masked_cfg)
    return baseline_results, masked_results
