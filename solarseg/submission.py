"""CSV contract validation, including every image and non-overlapping instances."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from pycocotools import mask as coco_mask

from solarseg.data import sha256
from solarseg.postprocess import instances


def write_submission(stems, cache, params, output):
    rows, counts = [], {}
    for stem in sorted(stems):
        rles = instances(np.load(Path(cache) / f"{stem}.npy"), **params)
        counts[stem] = len(rles)
        for i, rle in enumerate(rles, 1):
            rows.append(
                {"filament_id": f"{stem}_{i}", "segmentation_rle": rle["counts"].decode("ascii")}
            )
    output = Path(output)
    pd.DataFrame(rows, columns=["filament_id", "segmentation_rle"]).to_csv(output, index=False)
    # Zero detections are explicit in the sidecar, not fake empty-mask CSV rows.
    output.with_suffix(".images.json").write_text(json.dumps(counts, indent=2))
    return validate_submission(output, stems)


def validate_submission(path, expected_stems, shape=(2048, 2048)):
    path = Path(path)
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    if list(df.columns) != ["filament_id", "segmentation_rle"]:
        raise ValueError("CSV must contain exactly filament_id,segmentation_rle")
    if df.empty or df.filament_id.duplicated().any():
        raise ValueError("Submission is empty or contains duplicate filament IDs")
    ids = df.filament_id.str.rsplit("_", n=1, expand=True)
    if ids.shape[1] != 2 or not ids[1].str.fullmatch(r"[1-9][0-9]*").all():
        raise ValueError("Invalid filament ID suffix")
    expected_stems = set(expected_stems)
    if set(ids[0]) - expected_stems:
        raise ValueError("Unknown image prefix")
    counts = json.loads(path.with_suffix(".images.json").read_text())
    if set(counts) != expected_stems:
        raise ValueError("Inference coverage manifest is incomplete")
    actual = ids[0].value_counts().to_dict()
    if any(counts[k] != actual.get(k, 0) for k in counts):
        raise ValueError("CSV and coverage manifest disagree")
    for stem, group in df.groupby(ids[0]):
        claimed = np.zeros(shape, bool)
        seen = set()
        for text in group.segmentation_rle:
            if not text or text[0] in "\"'" or text[-1] in "\"'":
                raise ValueError("Empty or manually quoted RLE counts")
            if text in seen:
                raise ValueError("Duplicate predicted mask")
            seen.add(text)
            mask = coco_mask.decode({"size": list(shape), "counts": text.encode("ascii")}).astype(
                bool
            )
            if mask.shape != shape or not mask.any():
                raise ValueError("Empty mask or wrong dimensions")
            if np.any(mask & claimed):
                raise ValueError(f"Overlapping masks for {stem}")
            claimed |= mask
    result = {
        "valid": True,
        "rows": len(df),
        "images_processed": len(counts),
        "images_with_predictions": len(actual),
        "zero_detection_images": sum(v == 0 for v in counts.values()),
        "sha256": sha256(path),
    }
    path.with_suffix(".validation.json").write_text(json.dumps(result, indent=2))
    return result
