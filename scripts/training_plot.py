"""Plot recorded optimization histories without treating loss as validation PQ."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

root = Path(__file__).resolve().parents[1]
fig, ax = plt.subplots(figsize=(9, 4.3), layout="constrained")
for run, label, color in [
    ("unet-v1", "Width 16 · 1024 pixels · crop 256", "#466b9d"),
    ("unet-v2", "Width 24 · 1024 pixels · crop 384", "#a6611a"),
    ("unet-native-v1", "Width 24 · 2048 pixels · crop 384", "#21876d"),
]:
    frame = pd.DataFrame(json.loads((root / f"reports/training-history/{run}.json").read_text()))
    ax.plot(frame.step, frame.loss, color=color, alpha=.15, linewidth=.8)
    ax.plot(frame.step, frame.loss.rolling(5, min_periods=1).median(), color=color, label=label)
ax.set(xlabel="Optimizer updates", ylabel="Training BCE + soft Dice loss", ylim=(0, 2.1))
ax.set_title("Recorded training trajectories", loc="left", fontsize=14, fontweight="bold")
ax.text(.015, .97, "Faint: raw checkpoints\nSolid: median of 5 checkpoints",
        transform=ax.transAxes, fontsize=9, va="top")
ax.legend(frameon=False, fontsize=9)
ax.grid(alpha=.18)
ax.spines[["top", "right"]].set_visible(False)
fig.savefig(root / "reports/figures/training-curves.png", dpi=160)
plt.close(fig)
