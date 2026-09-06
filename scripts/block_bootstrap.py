"""Temporal-block uncertainty sensitivity from existing frozen holdout counts."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from solarseg.data import sha256
from solarseg.provenance import utc_now

root = Path(__file__).resolve().parents[1]
evaluation = root / "artifacts/unet-v2-tta/holdout.json"
manifest_path = root / "artifacts/manifests/train_manifest.csv"
record = json.loads(evaluation.read_text())
rows = pd.DataFrame(record["rows"])
manifest = pd.read_csv(manifest_path)
rows = rows.merge(manifest[["stem", "group", "role"]], on="stem", validate="many_to_one")
assert set(rows.role) == {"holdout"}
columns = ["iou_sum", "tp", "fp", "fn"]
groups = rows.groupby("group")[columns].sum()
values = groups.to_numpy()


def pq(counts):
    denominator = counts[..., 1] + 0.5 * (counts[..., 2] + counts[..., 3])
    return np.divide(
        counts[..., 0], denominator, out=np.zeros_like(denominator), where=denominator > 0
    )


assert np.isclose(pq(values.sum(axis=0)), record["pq"], rtol=0, atol=1e-12)
rng = np.random.default_rng(2026)
indices = rng.integers(0, len(groups), size=(5000, len(groups)))
samples = pq(values[indices].sum(axis=1))
result = {
    "created_at": utc_now(),
    "method": "Resample the original 27-day/duplicate-unioned holdout groups with replacement",
    "groups": len(groups),
    "physical_images": int(rows.stem.nunique()),
    "annotator_records": len(rows),
    "bootstrap_replicates": len(samples),
    "seed": 2026,
    "pq": record["pq"],
    "block_bootstrap_95ci": np.quantile(samples, [0.025, 0.975]).tolist(),
    "original_image_bootstrap_95ci": record["bootstrap_95ci"],
    "evaluation_sha256": sha256(evaluation),
    "manifest_sha256": sha256(manifest_path),
    "scope": "Post-evaluation uncertainty sensitivity only; no candidate or threshold selection",
    "limitation": "Fixed-model sampling uncertainty; recurring structures across groups may remain dependent",
}
output = root / "reports/temporal-bootstrap.json"
output.write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
