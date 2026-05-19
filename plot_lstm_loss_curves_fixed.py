"""
plot_lstm_loss_curves_fixed.py
-----------------------------
Plot real LSTM loss curves (train/val) from notebook memory or file.

Data source priority
--------------------
1) In-memory global variable `LSTM_LOSS_CURVES`
2) `--data` file path
3) Default files if present:
   - saved_models/lstm_loss_curves.json
   - saved_models/lstm_loss_curves.pkl
   - lstm_loss_curves.json
   - lstm_loss_curves.pkl

Accepted file formats
---------------------
- JSON/PKL with list[dict]: {"strategy": "...", "model": "...", "train_loss": [...], "val_loss": [...]}
- CSV with columns: run, epoch, train_loss[, val_loss]
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _normalize_raw_curves(raw: object) -> list[dict]:
    if not isinstance(raw, list):
        raise ValueError("Curves data must be a list of dict entries.")

    normalized = []
    for i, entry in enumerate(raw):
        if not isinstance(entry, dict):
            continue

        train_loss = entry.get("train_loss", [])
        val_loss = entry.get("val_loss", [])
        strategy = entry.get("strategy") or entry.get("name") or f"Run {i+1}"
        model = entry.get("model") or "LSTM"

        train_loss = list(train_loss) if train_loss is not None else []
        val_loss = list(val_loss) if val_loss is not None else []

        if len(train_loss) == 0 and len(val_loss) == 0:
            continue

        normalized.append(
            {
                "strategy": str(strategy),
                "model": str(model),
                "train_loss": train_loss,
                "val_loss": val_loss,
            }
        )

    if not normalized:
        raise ValueError("No valid curve entries found in data source.")
    return normalized


def _load_from_csv(path: Path) -> list[dict]:
    df = pd.read_csv(path)
    required = {"run", "epoch", "train_loss"}
    if not required.issubset(df.columns):
        raise ValueError(
            f"CSV must contain columns {sorted(required)}. Found: {list(df.columns)}"
        )

    curves = []
    df = df.sort_values(["run", "epoch"])
    for run, part in df.groupby("run"):
        entry = {
            "strategy": str(run),
            "model": str(part["model"].iloc[0]) if "model" in part.columns and len(part["model"]) > 0 else "LSTM",
            "train_loss": part["train_loss"].tolist(),
            "val_loss": part["val_loss"].tolist() if "val_loss" in part.columns else [],
        }
        curves.append(entry)
    return curves


def _load_from_file(path: Path) -> list[dict]:
    suffix = path.suffix.lower()
    if suffix == ".json":
        with path.open("r", encoding="utf-8") as f:
            return _normalize_raw_curves(json.load(f))
    if suffix in {".pkl", ".pickle"}:
        with path.open("rb") as f:
            return _normalize_raw_curves(pickle.load(f))
    if suffix == ".csv":
        return _normalize_raw_curves(_load_from_csv(path))
    raise ValueError(f"Unsupported data file type: {path.suffix}")


def resolve_curves(data_path: str | None = None) -> list[dict]:
    in_memory = globals().get("LSTM_LOSS_CURVES")
    if isinstance(in_memory, list) and len(in_memory) > 0:
        return _normalize_raw_curves(in_memory)

    candidates = []
    if data_path:
        candidates.append(Path(data_path))
    candidates.extend(
        [
            Path("saved_models/lstm_loss_curves.json"),
            Path("saved_models/lstm_loss_curves.pkl"),
            Path("lstm_loss_curves.json"),
            Path("lstm_loss_curves.pkl"),
        ]
    )

    for path in candidates:
        if path.exists() and path.is_file():
            print(f"[INFO] Loading real curves from: {path}")
            return _load_from_file(path)

    raise ValueError(
        "No real LSTM_LOSS_CURVES found. Provide --data <json|pkl|csv> or run this via notebook exec() after training."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot real LSTM loss curves")
    parser.add_argument("--data", type=str, default=None, help="Path to real curves file (.json/.pkl/.csv).")
    parser.add_argument("--out", type=str, default="lstm_loss_curves_fixed.png", help="Output image path.")
    args = parser.parse_args()

    curves = resolve_curves(args.data)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax_val = ax.twinx()
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    lines = []

    for i, curve in enumerate(curves):
        train_loss = curve.get("train_loss", [])
        val_loss = curve.get("val_loss", [])
        strategy = curve.get("strategy", curve.get("name", f"Run {i+1}"))
        model = curve.get("model", "LSTM")
        name = f"{strategy} ({model})"
        color = colors[i % len(colors)]

        if len(train_loss) > 0:
            x_train = np.arange(1, len(train_loss) + 1)
            line, = ax.plot(x_train, train_loss, color=color, linewidth=1.7, label=f"Training loss, {name}")
            lines.append(line)

        if len(val_loss) > 0 and not np.all(pd.isna(val_loss)):
            x_val = np.arange(1, len(val_loss) + 1)
            line, = ax_val.plot(
                x_val,
                val_loss,
                color=color,
                linestyle="--",
                linewidth=1.7,
                alpha=0.9,
                label=f"Validation loss, {name}",
            )
            lines.append(line)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training loss")
    ax_val.set_ylabel("Validation loss")
    ax.grid(True, color="#b0b0b0", linewidth=1, alpha=0.65)

    ax.legend(handles=lines, loc="upper right", frameon=True, framealpha=0.85, facecolor="white")

    fig.tight_layout()
    plt.savefig(args.out, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
