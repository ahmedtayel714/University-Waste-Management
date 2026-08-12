"""Parse Ultralytics results.csv from training runs and compare baseline vs
masked-preprocessing experiments — the headline evidence for the project."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

METRIC_COLS = {
    "precision": "metrics/precision(B)",
    "recall": "metrics/recall(B)",
    "mAP50": "metrics/mAP50(B)",
    "mAP50-95": "metrics/mAP50-95(B)",
}


def load_results_csv(run_dir: str) -> pd.DataFrame:
    csv_path = Path(run_dir) / "results.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"No results.csv in {run_dir} — has training finished?")
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    return df


def summarize_final_metrics(df: pd.DataFrame) -> dict:
    last = df.iloc[-1]
    return {name: float(last[col]) for name, col in METRIC_COLS.items() if col in df.columns}


def plot_loss_curves(df: pd.DataFrame, out_path: str, title: str = "Training / Validation Loss"):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    loss_pairs = [("box_loss", "train/box_loss", "val/box_loss"),
                  ("cls_loss", "train/cls_loss", "val/cls_loss"),
                  ("dfl_loss", "train/dfl_loss", "val/dfl_loss")]
    for ax, (label, train_col, val_col) in zip(axes, loss_pairs):
        if train_col in df.columns:
            ax.plot(df["epoch"], df[train_col], label="train")
        if val_col in df.columns:
            ax.plot(df["epoch"], df[val_col], label="val")
        ax.set_title(label)
        ax.set_xlabel("epoch")
        ax.legend()
    fig.suptitle(title)
    fig.tight_layout()
    _save(fig, out_path)


def plot_map_curves(df: pd.DataFrame, out_path: str, title: str = "mAP over training"):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for name, col in (("mAP50", METRIC_COLS["mAP50"]), ("mAP50-95", METRIC_COLS["mAP50-95"])):
        if col in df.columns:
            ax.plot(df["epoch"], df[col], label=name)
    ax.set_xlabel("epoch")
    ax.set_ylabel("mAP")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    _save(fig, out_path)


def compare_runs(baseline_dir: str, masked_dir: str, out_path: str) -> dict:
    baseline_df = load_results_csv(baseline_dir)
    masked_df = load_results_csv(masked_dir)
    baseline_final = summarize_final_metrics(baseline_df)
    masked_final = summarize_final_metrics(masked_df)

    labels = list(METRIC_COLS.keys())
    baseline_vals = [baseline_final.get(k, 0.0) for k in labels]
    masked_vals = [masked_final.get(k, 0.0) for k in labels]

    x = range(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar([i - width / 2 for i in x], baseline_vals, width, label="baseline (unmasked)")
    ax.bar([i + width / 2 for i in x], masked_vals, width, label="HSV-masked")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.0)
    ax.set_title("Baseline vs HSV-Masked Preprocessing — Final Metrics")
    ax.legend()
    for i, (b, m) in enumerate(zip(baseline_vals, masked_vals)):
        ax.text(i - width / 2, b + 0.01, f"{b:.3f}", ha="center", fontsize=8)
        ax.text(i + width / 2, m + 0.01, f"{m:.3f}", ha="center", fontsize=8)
    fig.tight_layout()
    _save(fig, out_path)

    return {"baseline": baseline_final, "masked": masked_final}


def _save(fig, out_path: str):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
