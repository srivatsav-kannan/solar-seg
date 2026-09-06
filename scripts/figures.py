"""Generate publication figures from measured calibration/holdout evidence."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from PIL import Image
from scipy import ndimage as ndi

from solarseg.data import CompetitionData
from solarseg.engine import evaluate_and_save
from solarseg.metrics import aggregate, bootstrap_pq
from solarseg.postprocess import decode, instances

p = argparse.ArgumentParser()
p.add_argument("--run", required=True)
p.add_argument("--data", default="data/raw")
a = p.parse_args()
run = Path(a.run)
out = Path("reports/figures")
out.mkdir(parents=True, exist_ok=True)
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)
data = CompetitionData(a.data)
manifest = pd.read_csv("artifacts/manifests/train_manifest.csv")
params = json.loads((run / "postprocess.json").read_text())
cal_path = run / "calibration-selected.json"
if not cal_path.exists():
    evaluate_and_save(
        data,
        manifest.loc[manifest.role == "calibration", "stem"],
        run / "probabilities-calibration",
        params,
        cal_path,
    )
cal = json.loads(cal_path.read_text())

# Exact implemented workflow; no proposed architecture is shown as implemented.
fig, ax = plt.subplots(figsize=(12, 3.1), layout="constrained")
ax.set(xlim=(0, 12), ylim=(0, 3))
ax.axis("off")
labels = [
    (
        0.1,
        1.8,
        "Official JPEGs + labels",
        "707 physical training images\nAnnotators stay in one fold",
    ),
    (4.15, 1.8, "Grouped validation", "399 train / 149 calibrate\n145 holdout / 14 embargo"),
    (8.2, 1.8, "Learn foreground", "One-channel U-Net\nBCE + soft Dice, no pretraining"),
    (8.2, 0.1, "Predict probabilities", "1024-pixel disk\nRestore to 2048 before threshold"),
    (
        4.15,
        0.1,
        "Recover instances",
        "Calibrated threshold + closing\nDisjoint connected components",
    ),
    (0.1, 0.1, "Evaluate and submit", "Pooled per-annotator PQ\nValidated compressed COCO RLE"),
]
for x, y, title, text in labels:
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            3.65,
            1.12,
            boxstyle="round,pad=0.06",
            linewidth=1,
            edgecolor="#466779",
            facecolor="#eef5f6",
        )
    )
    ax.text(
        x + 1.825,
        y + 0.83,
        title,
        ha="center",
        va="center",
        weight="bold",
        color="#183f52",
        fontsize=13,
    )
    ax.text(x + 1.825, y + 0.36, text, ha="center", va="center", fontsize=12)
for start, end in [
    ((3.85, 2.36), (4.05, 2.36)),
    ((7.9, 2.36), (8.1, 2.36)),
    ((10.0, 1.72), (10.0, 1.3)),
    ((8.12, 0.66), (7.93, 0.66)),
    ((4.07, 0.66), (3.88, 0.66)),
]:
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=15, color="#466779"))
fig.savefig(out / "pipeline.png", dpi=190, bbox_inches="tight")
fig.savefig(out / "pipeline.svg", bbox_inches="tight")
plt.close(fig)

# Systematic selection: worst, middle, and best physical-image PQ among images
# with at least three GT instances. Use the first supplied annotator for display.
grouped = {}
for row in cal["rows"]:
    grouped.setdefault(row["stem"], []).append(row)
ranking = sorted(
    [
        (aggregate(rows)["pq"], stem)
        for stem, rows in grouped.items()
        if sum(r["n_gt"] for r in rows) >= 3
    ]
)
chosen = [ranking[0], ranking[len(ranking) // 2], ranking[-1]]
fig, axes = plt.subplots(3, 3, figsize=(10.5, 10.5), layout="constrained")
example_records = []
for row_idx, ((pq, stem), label) in enumerate(
    zip(chosen, ["Worst", "Middle", "Best"], strict=True)
):
    image = np.asarray(Image.open(data.image_path(stem)).convert("L"))
    annotator, gt = next(iter(data.ground_truth(stem).items()))
    masks = [decode(r) for r in gt]
    pred = [
        decode(r)
        for r in instances(np.load(run / "probabilities-calibration" / f"{stem}.npy"), **params)
    ]
    largest = max(masks, key=lambda x: x.sum())
    yy, xx = np.where(largest)
    size = max(640, int(max(yy.max() - yy.min(), xx.max() - xx.min())) + 180)
    size = min(size, 1600)
    y = int(np.clip((yy.max() + yy.min() - size) // 2, 0, 2048 - size))
    x = int(np.clip((xx.max() + xx.min() - size) // 2, 0, 2048 - size))
    for col in range(3):
        ax = axes[row_idx, col]
        ax.imshow(image[y : y + size, x : x + size], cmap="gray", vmin=0, vmax=255)
        if col:
            overlay = np.zeros((size, size, 4))
            for mask in masks if col == 1 else pred:
                boundary = ndi.binary_dilation(mask ^ ndi.binary_erosion(mask), iterations=1)
                overlay[boundary[y : y + size, x : x + size]] = (
                    [0.0, 0.95, 0.8, 1] if col == 1 else [1, 0.35, 0.65, 1]
                )
            ax.imshow(overlay)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(
            ["H-alpha input", "One annotator (cyan)", "Prediction (pink)"][col], fontsize=10
        )
    axes[row_idx, 0].set_ylabel(f"{label} case; image PQ {pq:.3f}\n{stem}", fontsize=9)
    example_records.append(
        {
            "stem": stem,
            "selection": label.lower(),
            "pq": pq,
            "annotator_displayed": annotator,
            "crop": [x, y, size, size],
        }
    )
fig.savefig(out / "calibration-examples.png", dpi=150, bbox_inches="tight")
plt.close(fig)
(run / "visual-examples.json").write_text(json.dumps(example_records, indent=2))

holdout_path = run / "holdout.json"
if holdout_path.exists():
    h = json.loads(holdout_path.read_text())
    plt.rcParams.update({"font.size": 12, "axes.titlesize": 11})
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), layout="constrained")
    by_site = {
        s: [r for r in h["rows"] if r["stem"].endswith(s)] for s in sorted(manifest.site.unique())
    }
    sites = list(by_site)
    scores = np.array([aggregate(by_site[s])["pq"] for s in sites])
    intervals = np.array([bootstrap_pq(by_site[s]) for s in sites])
    axes[0].bar(sites, scores, color="#317c90")
    axes[0].errorbar(
        sites,
        scores,
        yerr=np.maximum(0, np.stack([scores - intervals[:, 0], intervals[:, 1] - scores])),
        fmt="none",
        color="#173f52",
        capsize=3,
    )
    axes[0].set(
        ylabel="Pooled PQ",
        title="Holdout by site; physical-image 95% CI",
        ylim=(0, max(0.5, intervals.max() + 0.02)),
    )
    for key, label in [("positive_iou", "IoU"), ("positive_dice", "Dice")]:
        values = [v for row in h["rows"] for v in row[key]]
        axes[1].hist(
            values, bins=np.linspace(0, 1, 21), histtype="step", linewidth=1.6, label=label
        )
    axes[1].set(
        xlabel="Overlap (positive pairs only)",
        ylabel="Pair count",
        title="Misses remain in FP/FN counts",
    )
    axes[1].legend(frameon=False)
    names = ["small", "medium", "large"]
    totals = [sum(r["size_bins"][k]["gt"] for r in h["rows"]) for k in names]
    recalls = [
        sum(r["size_bins"][k]["matched"] for r in h["rows"]) / max(n, 1)
        for k, n in zip(names, totals, strict=True)
    ]
    axes[2].bar(
        [f"{k}\nn={n}" for k, n in zip(names, totals, strict=True)], recalls, color="#cd744f"
    )
    axes[2].set(
        ylabel="GT recall at IoU > 0.5", ylim=(0, 1), title="Area bins: <1k / 1k–10k / ≥10k px"
    )
    fig.savefig(out / "holdout-diagnostics.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    summary = {k: v for k, v in h.items() if k != "rows"}
    summary["by_site"] = {
        s: dict(**aggregate(rows), bootstrap_95ci=bootstrap_pq(rows)) for s, rows in by_site.items()
    }
    summary["by_year"] = {
        y: aggregate([r for r in h["rows"] if r["stem"].startswith(y)])
        for y in sorted({r["stem"][:4] for r in h["rows"]})
    }
    summary["size_recall"] = {
        k: {"gt": n, "recall": r} for k, n, r in zip(names, totals, recalls, strict=True)
    }
    Path("reports/holdout-summary.json").write_text(json.dumps(summary, indent=2))
print(f"Figures written to {out}")
