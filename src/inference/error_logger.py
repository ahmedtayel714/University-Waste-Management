"""Low-confidence detection logging for active learning. Saves an annotated
snapshot plus structured metadata whenever a detection falls below a
confidence threshold, so uncertain cases accumulate into a ready-made
relabeling queue instead of disappearing silently."""

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2

from .predict import Detection, draw_detections


@dataclass
class LoggedCase:
    source_image: str
    saved_snapshot: str
    timestamp: str
    min_confidence: float
    detections: list


class LowConfidenceLogger:
    """Call `log()` once per inference; it no-ops unless at least one
    detection falls under `threshold`. Every logged case gets an annotated
    snapshot (so a human can eyeball it fast) and an entry appended to a
    single JSONL index file (so the whole log is scriptable later — e.g.
    feed straight into a relabeling tool)."""

    def __init__(self, out_dir: Path, threshold: float = 0.5, class_names: list = None):
        self.out_dir = Path(out_dir)
        self.snapshot_dir = self.out_dir / "snapshots"
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        self.threshold = threshold
        self.class_names = class_names or ["leaf"]
        self.index_path = self.out_dir / "low_confidence_log.jsonl"

    def log(self, image, detections: list, source_name: str) -> LoggedCase | None:
        low_conf = [d for d in detections if d.conf < self.threshold]
        if not low_conf:
            return None

        stem = Path(source_name).stem
        snapshot_name = f"{stem}_{int(time.time() * 1000)}.jpg"
        snapshot_path = self.snapshot_dir / snapshot_name
        annotated = draw_detections(image, detections, self.class_names)
        cv2.imwrite(str(snapshot_path), annotated)

        case = LoggedCase(
            source_image=str(source_name),
            saved_snapshot=str(snapshot_path),
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
            min_confidence=round(min(d.conf for d in low_conf), 4),
            detections=[
                {"box": list(d.box), "confidence": round(d.conf, 4), "class": d.cls}
                for d in detections
            ],
        )
        with open(self.index_path, "a") as f:
            f.write(json.dumps(asdict(case)) + "\n")
        return case

    def summary(self) -> dict:
        """Quick counts without re-parsing every snapshot — how many cases
        logged, and the confidence range that triggered them."""
        if not self.index_path.exists():
            return {"cases": 0}
        lines = [json.loads(l) for l in self.index_path.read_text().splitlines() if l.strip()]
        if not lines:
            return {"cases": 0}
        min_confs = [c["min_confidence"] for c in lines]
        return {
            "cases": len(lines),
            "lowest_confidence": min(min_confs),
            "highest_of_the_low": max(min_confs),
        }
