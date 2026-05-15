"""
plot_mlp_loss_curves.py
-----------------------
Fixed MLP training loss curve plot with:
  - Dict-based grouping (no silent label misassignment)
  - Controlled y-axis so Progressive Unfreezing shading doesn't dominate
  - Reduced std shading (x0.15)
  - Annotation for warm-initialized strategies
  - Convergence check warning if curves are still descending at the end
  - Epoch x-axis matches actual data length (not hardcoded 10)

Usage
-----
  In your Jupyter notebook 03_mlp.ipynb, run this as a cell:
  
      exec(open('plot_mlp_loss_curves_fixed.py').read())
  
  Or standalone (paste your MLP_LOSS_CURVES list into this file):
  
      python plot_mlp_loss_curves_fixed.py

MLP_LOSS_CURVES format expected
--------------------------------
Each element must be a dict with at minimum:
    {
        "strategy": "FTL",          # must match one of STRATEGY_NAMES
        "train_loss": [0.9, 0.8, …] # list of per-epoch loss values
    }
"""

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


# ── helpers ───────────────────────────────────────────────────────────────────

def smooth(values: np.ndarray, weight: float = SMOOTH_W) -> np.ndarray:
    """Exponential moving average smoothing."""
    if len(values) == 0:
        return np.array([])
    out, last = [], float(values[0])
    for v in values:
        last = last * weight + (1 - weight) * v
        out.append(last)
    return np.array(out)


def pad_stack(curves: list) -> np.ndarray:
    """Pad curves to same length and stack."""
    max_len = max(len(c) for c in curves)
    padded = []
    for c in curves:
        arr = np.array(c, dtype=float)
        if len(arr) < max_len:
            arr = np.concatenate([arr, np.full(max_len - len(arr), arr[-1])])
        padded.append(arr)
    return np.stack(padded)


def aggregate(curves: list) -> dict | None:
    """Return mean ± std arrays for a list of loss-curve lists."""
    valid = [c for c in curves if len(c) > 0]
    if not valid:
        return None
    mat = pad_stack(valid)
    return {"mean": mat.mean(axis=0), "std": mat.std(axis=0)}


def is_still_descending(curve: np.ndarray, tail: float = 0.3) -> bool:
    """True if the last `tail` fraction of the curve is still going down."""
    n = len(curve)
    tail_seg = curve[int(n * (1 - tail)):]
    return tail_seg[-1] < tail_seg[0] * 0.97   # still dropping > 3%


# ── group curves by strategy name (safe, no silent misassignment) ─────────────

grouped: dict[str, list] = {s: [] for s in STRATEGY_NAMES}

for entry in MLP_LOSS_CURVES:
    name = entry.get("strategy", "")
    loss = entry.get("train_loss", [])
    if name in grouped and len(loss) > 0:
        grouped[name].append(loss)
    elif name not in grouped:
        print(f"[WARN] Unknown strategy '{name}' — skipped.")

# ── convergence check ─────────────────────────────────────────────────────────

for name, curves in grouped.items():
    agg = aggregate(curves)
    if agg is None:
        continue
    if is_still_descending(agg["mean"]):
        print(f"[WARN] '{name}' mean loss still descending at final epoch — "
              "consider more epochs or verify data.")

# ── plot ──────────────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(11, 6.5))

handles = []
warm_y_positions = []

for name in STRATEGY_NAMES:
    curves = grouped[name]
    agg = aggregate(curves)
    if agg is None:
        print(f"[INFO] No data for strategy '{name}' — skipped.")
        continue

    y      = smooth(agg["mean"])
    y_std  = agg["std"] * STD_SCALE
    x      = np.arange(1, len(y) + 1)
    color  = COLORS[name]

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

# ── warm-init annotation ──────────────────────────────────────────────────────

if warm_y_positions:
    annotation_y = min(warm_y_positions) - 0.12
    ax.annotate(
        "↑ Warm init (pretrained start)",
        xy=(1.15, min(warm_y_positions)),
        xytext=(2.5, annotation_y + 0.35),
        fontsize=8.5,
        color="#555555",
        arrowprops=dict(arrowstyle="-|>", color="#888888", lw=0.9),
        va="top",
    )

# ── axes & styling ────────────────────────────────────────────────────────────

max_epochs = max(
    len(c) for curves in grouped.values() for c in curves
) if any(grouped.values()) else 10

ax.set_xlim(1, max_epochs)
ax.set_ylim(Y_MIN, Y_MAX)

ax.set_title("MLP Training Loss Curves (All Strategies)", fontsize=15, pad=14, fontweight='bold')
ax.set_xlabel("Epoch", fontsize=13, fontweight='bold')
ax.set_ylabel("Training Loss (MSE)", fontsize=13, fontweight='bold')

ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
ax.grid(True, alpha=0.3, linestyle="--")

# Light horizontal reference at y=1.0 (rough "acceptable" threshold marker)
ax.axhline(1.0, color="black", linewidth=0.6, linestyle=":", alpha=0.4)
ax.text(max_epochs - 0.1, 1.02, "loss = 1.0", fontsize=7.5,
        color="#777777", ha="right")

ax.legend(
    handles=handles,
    loc="upper center",
    bbox_to_anchor=(0.5, -0.14),
    ncol=3,
    frameon=False,
    fontsize=10,
)

plt.tight_layout()
plt.show()

print(f"\n✓ Plotted {len(handles)} strategies with {len([c for curves in grouped.values() for c in curves])} total loss curves")
