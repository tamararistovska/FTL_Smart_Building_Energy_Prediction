"""
plot_mlp_loss_curves_fixed.py
-----------------------------
Plot real MLP training loss curves (mean +/- scaled std) by strategy.

Data source priority
--------------------
1) In-memory global variable `MLP_LOSS_CURVES` (best for notebook `exec(...)`)
2) `--data` path passed on CLI
3) Default files if present:
   - saved_models/mlp_loss_curves.json
   - saved_models/mlp_loss_curves.pkl
   - mlp_loss_curves.json
   - mlp_loss_curves.pkl

Accepted file formats
---------------------
- JSON/PKL with list[dict]: {"strategy": "...", "model": "...", "train_loss": [...]}
- CSV with columns: strategy, epoch, train_loss
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

STRATEGY_NAMES = [
    "FTL",
    "Personalized-FL",
    "Progressive Unfreezing",
    "Instance-TL",
    "Fed-SimTL",
    "FedMetaTL",
]

WARM_INIT_STRATEGIES = {"FTL", "Personalized-FL"}

COLORS = {
    "FTL": "#1f77b4",
    "Personalized-FL": "#ff7f0e",
    "Progressive Unfreezing": "#2ca02c",
    "Instance-TL": "#d62728",
    "Fed-SimTL": "#9467bd",
    "FedMetaTL": "#8c564b",
}

STD_SCALE = 0.15
SMOOTH_W = 0.75
Y_MAX = 3.6
Y_MIN = 0.0


def smooth(values: np.ndarray, weight: float = SMOOTH_W) -> np.ndarray:
    if len(values) == 0:
        return np.array([])
    out, last = [], float(values[0])
    for v in values:
        last = last * weight + (1 - weight) * v
        out.append(last)
    return np.array(out)


def pad_stack(curves: list[list[float]]) -> np.ndarray:
    max_len = max(len(c) for c in curves)
    padded = []
    for c in curves:
        arr = np.array(c, dtype=float)
        if len(arr) < max_len:
            arr = np.concatenate([arr, np.full(max_len - len(arr), arr[-1])])
        padded.append(arr)
    return np.stack(padded)


def aggregate(curves: list[list[float]]) -> dict[str, np.ndarray] | None:
    valid = [c for c in curves if len(c) > 0]
    if not valid:
        return None
    mat = pad_stack(valid)
    return {"mean": mat.mean(axis=0), "std": mat.std(axis=0)}


def is_still_descending(curve: np.ndarray, tail: float = 0.3) -> bool:
    n = len(curve)
    tail_seg = curve[int(n * (1 - tail)) :]
    return tail_seg[-1] < tail_seg[0] * 0.97


def _normalize_raw_curves(raw: object) -> list[dict]:
    if not isinstance(raw, list):
        raise ValueError("Curves data must be a list of dict entries.")

    normalized = []
    for i, entry in enumerate(raw):
        if not isinstance(entry, dict):
            continue
        strategy = entry.get("strategy") or entry.get("name")
        model = entry.get("model") or "MLP"
        train_loss = entry.get("train_loss")
        if strategy is None or train_loss is None:
            continue
        train_loss = list(train_loss)
        if len(train_loss) == 0:
            continue
        normalized.append(
            {"strategy": str(strategy), "model": str(model), "train_loss": train_loss}
        )

    if not normalized:
        raise ValueError("No valid curve entries found in data source.")
    return normalized


def _load_from_csv(path: Path) -> list[dict]:
    df = pd.read_csv(path)
    required = {"strategy", "epoch", "train_loss"}
    if not required.issubset(df.columns):
        raise ValueError(
            f"CSV must contain columns {sorted(required)}. Found: {list(df.columns)}"
        )
    df = df.sort_values(["strategy", "epoch"])
    curves = []
    for strategy, part in df.groupby("strategy"):
        model = (
            str(part["model"].iloc[0])
            if "model" in part.columns and len(part["model"]) > 0
            else "MLP"
        )
        curves.append(
            {"strategy": strategy, "model": model, "train_loss": part["train_loss"].tolist()}
        )
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
    in_memory = globals().get("MLP_LOSS_CURVES")
    if isinstance(in_memory, list) and len(in_memory) > 0:
        return _normalize_raw_curves(in_memory)

    candidates = []
    if data_path:
        candidates.append(Path(data_path))
    candidates.extend(
        [
            Path("saved_models/mlp_loss_curves.json"),
            Path("saved_models/mlp_loss_curves.pkl"),
            Path("mlp_loss_curves.json"),
            Path("mlp_loss_curves.pkl"),
        ]
    )

    for path in candidates:
        if path.exists() and path.is_file():
            print(f"[INFO] Loading real curves from: {path}")
            return _load_from_file(path)

    raise ValueError(
        "No real MLP_LOSS_CURVES found. Provide --data <json|pkl|csv> or run this via notebook exec() after training."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot real MLP loss curves")
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Path to real curves file (.json/.pkl/.csv).",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="mlp_loss_curves_fixed.png",
        help="Output image file path.",
    )
    args = parser.parse_args()

    curves_source = resolve_curves(args.data)

    observed_names = []
    for entry in curves_source:
        name = entry.get("strategy") or entry.get("name") or "Unknown"
        if name not in observed_names:
            observed_names.append(name)

    plot_order = [s for s in STRATEGY_NAMES if s in observed_names]
    plot_order += [s for s in observed_names if s not in plot_order]

    grouped: dict[str, list[list[float]]] = {s: [] for s in plot_order}

    for entry in curves_source:
        name = entry.get("strategy") or entry.get("name") or "Unknown"
        loss = entry.get("train_loss", [])
        if name in grouped and len(loss) > 0:
            grouped[name].append(loss)
        elif name not in grouped:
            print(f"[WARN] Unknown strategy '{name}' - skipped.")

    for name, curves in grouped.items():
        agg = aggregate(curves)
        if agg is None:
            continue
        if is_still_descending(agg["mean"]):
            print(
                f"[WARN] '{name}' mean loss still descending at final epoch - "
                "consider more epochs or verify data."
            )

    fig, ax = plt.subplots(figsize=(10, 6))

    handles = []
    warm_y_positions = []

    fallback_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for i, name in enumerate(plot_order):
        curves = grouped[name]
        agg = aggregate(curves)
        if agg is None:
            print(f"[INFO] No data for strategy '{name}' - skipped.")
            continue

        y = smooth(agg["mean"])
        y_std = agg["std"] * STD_SCALE
        x = np.arange(1, len(y) + 1)
        color = COLORS.get(name, fallback_colors[i % len(fallback_colors)])

        line, = ax.plot(x, y, linewidth=2.2, color=color, label=name)

        ax.fill_between(
            x,
            np.clip(y - y_std, Y_MIN, Y_MAX),
            np.clip(y + y_std, Y_MIN, Y_MAX),
            color=color,
            alpha=0.12,
        )

        handles.append(line)

        if name in WARM_INIT_STRATEGIES:
            warm_y_positions.append(float(y[0]))

    if warm_y_positions:
        annotation_y = min(warm_y_positions) - 0.12
        ax.annotate(
            "? Warm init (pretrained start)",
            xy=(1.15, min(warm_y_positions)),
            xytext=(2.5, annotation_y + 0.35),
            fontsize=8.5,
            color="#555555",
            arrowprops=dict(arrowstyle="-|>", color="#888888", lw=0.9),
            va="top",
        )

    max_epochs = (
        max(len(c) for curves in grouped.values() for c in curves)
        if any(grouped.values())
        else 10
    )

    ax.set_xlim(1, max_epochs)
    ax.set_ylim(Y_MIN, Y_MAX)

    ax.set_title("MLP Training Loss Curves", fontsize=15, pad=14)
    ax.set_xlabel("Epoch", fontsize=13)
    ax.set_ylabel("Training Loss (MSE)", fontsize=13)

    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.grid(True, alpha=0.3, linestyle="--")

    ax.axhline(1.0, color="black", linewidth=0.6, linestyle=":", alpha=0.4)
    ax.text(max_epochs - 0.1, 1.02, "loss = 1.0", fontsize=7.5, color="#777777", ha="right")

    ax.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.14),
        ncol=3,
        frameon=False,
        fontsize=10,
    )

    plt.tight_layout()
    plt.savefig(args.out, dpi=300, bbox_inches="tight")
    plt.show()

    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
