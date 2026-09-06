"""Describe calibration-label disagreement; this is not a model score ceiling."""

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from pycocotools import mask as coco_mask

from solarseg.data import CompetitionData, sha256
from solarseg.metrics import aggregate, counts, overlap
from solarseg.provenance import utc_now

root = Path(__file__).resolve().parents[1]
data = CompetitionData(root / "data/raw")
manifest_path = root / "artifacts/manifests/train_manifest.csv"
manifest = pd.read_csv(manifest_path)
rows = []
for stem in manifest.loc[manifest.role == "calibration", "stem"]:
    entries = data.ground_truth(stem)
    for (first_id, first), (second_id, second) in itertools.combinations(entries.items(), 2):
        instance_pq = aggregate([counts(overlap(first, second))])["pq"]
        if first and second:
            first_union, second_union = coco_mask.merge(first), coco_mask.merge(second)
            intersection = float(coco_mask.area(coco_mask.merge([first_union, second_union], True)))
            denominator = float(coco_mask.area(first_union)) + float(coco_mask.area(second_union))
            union_dice = 2 * intersection / denominator
        else:
            union_dice = 1.0 if not first and not second else 0.0
        rows.append({
            "stem": stem,
            "first_annotator_image": first_id,
            "second_annotator_image": second_id,
            "first_instances": len(first),
            "second_instances": len(second),
            "instance_pq": instance_pq,
            "foreground_dice": union_dice,
        })
frame = pd.DataFrame(rows)
physical = frame.groupby("stem")[["instance_pq", "foreground_dice"]].mean()
assert np.isfinite(physical.to_numpy()).all()
result = {
    "created_at": utc_now(),
    "role": "calibration",
    "physical_images_with_multiple_annotators": len(physical),
    "annotator_pairs": len(frame),
    "mean_pairwise_pq_per_physical_image": float(physical.instance_pq.mean()),
    "median_pairwise_pq_per_physical_image": float(physical.instance_pq.median()),
    "mean_pairwise_foreground_dice_per_physical_image": float(physical.foreground_dice.mean()),
    "pq_physical_image_quartiles": physical.instance_pq.quantile([0.25, 0.5, 0.75]).tolist(),
    "annotation_sha256": sha256(data.annotation_path),
    "manifest_sha256": sha256(manifest_path),
    "scope": "Pairwise annotator agreement diagnostic, equal weight per physical calibration image",
    "limitations": [
        "Not a model performance ceiling or a competition score; pairwise comparisons differ from the full evaluator's pooling",
        "Annotations can overlap internally, unlike required predicted masks",
        "Only observations with multiple annotation sets contribute",
    ],
}
frame.to_csv(root / "reports/annotator-pairs.csv", index=False)
(root / "reports/annotator-agreement.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
