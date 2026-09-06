"""Evaluate the unchanged grid while reusing components across area cutoffs."""

import json
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from pycocotools import mask as coco_mask
from tqdm.auto import tqdm

from solarseg.engine import calibration_grid
from solarseg.metrics import aggregate, evaluate_image
from solarseg.postprocess import instances


def image_grid(data, stem, probability, grid, native_size=2048):
    groups = defaultdict(list)
    for i, params in enumerate(grid):
        groups[(params["threshold"], params["closing"])].append(i)
    rows = [None] * len(grid)
    for (threshold, closing), indices in groups.items():
        minimum = min(grid[i]["min_area"] for i in indices)
        components = instances(
            probability,
            threshold=threshold,
            closing=closing,
            min_area=minimum,
            native_size=native_size,
        )
        areas = [int(coco_mask.area(rle)) for rle in components]
        for i in indices:
            retained = [
                rle
                for rle, area in zip(components, areas, strict=True)
                if area >= grid[i]["min_area"]
            ]
            rows[i] = evaluate_image(data, stem, retained)
    return rows


def calibrate(data, stems, cache, out_dir, workers=4):
    out = Path(out_dir)
    if (out / "calibration.json").exists():
        raise FileExistsError("Completed calibration is immutable")
    grid = calibration_grid()
    combined = [[] for _ in grid]

    def one(stem):
        return image_grid(data, stem, np.load(Path(cache) / f"{stem}.npy"), grid)

    # map preserves stem order, including the floating-point summation order.
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for rows in tqdm(pool.map(one, stems), total=len(stems), desc="Calibration grid"):
            for target, additions in zip(combined, rows, strict=True):
                target.extend(additions)
    results = [
        dict(params=params, **aggregate(rows)) for params, rows in zip(grid, combined, strict=True)
    ]
    best_index = max(range(len(results)), key=lambda i: results[i]["pq"])
    best = results[best_index]
    out.mkdir(parents=True, exist_ok=True)
    (out / "calibration.json").write_text(json.dumps(results, indent=2))
    (out / "postprocess.json").write_text(json.dumps(best["params"], indent=2))
    for result in results:
        print(json.dumps(result), flush=True)
    return best, combined[best_index]
