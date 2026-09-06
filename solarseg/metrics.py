"""RLE-efficient reproduction of the organizers' pooled, per-annotator PQ."""

from __future__ import annotations

import numpy as np
from pycocotools import mask as coco_mask


def overlap(gt, pred):
    if not gt or not pred:
        return np.zeros((len(gt), len(pred)), dtype=np.float64)
    # COCO's crowd flag changes the denominator; always false for this challenge.
    return np.asarray(coco_mask.iou(pred, gt, [0] * len(gt))).T


def counts(iou):
    hit = iou > 0.5  # Strict >, as in the official notebook.
    return {
        "tp": int(hit.sum()),
        "fp": int((hit.sum(0) == 0).sum()),
        "fn": int((hit.sum(1) == 0).sum()),
        "iou_sum": float(iou[hit].sum()),
    }


def aggregate(rows):
    total = {k: sum(r[k] for r in rows) for k in ["tp", "fp", "fn", "iou_sum"]}
    den = total["tp"] + 0.5 * (total["fp"] + total["fn"])
    total["pq"] = total["iou_sum"] / den if den else 0.0
    total["sq"] = total["iou_sum"] / total["tp"] if total["tp"] else 0.0
    total["rq"] = total["tp"] / den if den else 0.0
    return total


def evaluate_image(data, stem, predictions):
    rows = []
    for annotator_image, gt in data.ground_truth(stem).items():
        iou = overlap(gt, predictions)
        row = dict(stem=stem, annotator_image=annotator_image, **counts(iou))
        row["fragmented_gt"] = int(((iou > 0).sum(1) > 1).sum())
        row["merged_pred"] = int(((iou > 0).sum(0) > 1).sum())
        row["n_gt"], row["n_pred"] = iou.shape
        areas = np.array([int(coco_mask.area(rle)) for rle in gt])
        matched = (iou > 0.5).any(axis=1)
        row["size_bins"] = {}
        for name, lo, hi in [("small", 0, 1000), ("medium", 1000, 10000), ("large", 10000, np.inf)]:
            selected = (areas >= lo) & (areas < hi)
            row["size_bins"][name] = {
                "gt": int(selected.sum()),
                "matched": int((selected & matched).sum()),
            }
        row["positive_iou"] = iou[iou > 0].tolist()
        row["positive_dice"] = (2 * iou[iou > 0] / (1 + iou[iou > 0])).tolist()
        rows.append(row)
    return rows


def bootstrap_pq(rows, n=1000, seed=2026):
    """Resample physical observations, retaining their annotators as a cluster."""
    grouped = {}
    for row in rows:
        grouped.setdefault(row["stem"], []).append(row)
    values = list(grouped.values())
    if not values:
        return [0.0, 0.0]
    rng = np.random.default_rng(seed)
    scores = [
        aggregate([r for i in rng.integers(0, len(values), len(values)) for r in values[i]])["pq"]
        for _ in range(n)
    ]
    return np.quantile(scores, [0.025, 0.975]).tolist()
