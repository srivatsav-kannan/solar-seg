"""Freeze the first campaign using calibration only, before holdout inspection."""

import json
from pathlib import Path

from solarseg.data import sha256
from solarseg.provenance import source_hashes, utc_now

definitions = [
    ("unet-v1", "unet-v1", 0, False),
    ("unet-v1-tiled", "unet-v1", 256, False),
    ("unet-v2", "unet-v2", 0, False),
    ("unet-v2-tta", "unet-v2", 0, True),
]
candidates = []
for name, training_name, tile, tta in definitions:
    run = Path("artifacts") / name
    values = json.loads((run / "calibration.json").read_text())
    expanded = run / "calibration-expanded.json"
    if expanded.exists():
        values += json.loads(expanded.read_text())
    best = max(values, key=lambda v: v["pq"])
    candidates.append(
        {
            "run": str(run),
            "training_run": "artifacts/" + training_name,
            "tile": tile,
            "tta": tta,
            "calibration_pq": best["pq"],
            "postprocess": best["params"],
        }
    )
ordered = sorted(candidates, key=lambda x: x["calibration_pq"], reverse=True)
selected = ordered[0].copy()
if selected["tta"] and selected["calibration_pq"] - ordered[1]["calibration_pq"] < 0.005:
    selected = ordered[1].copy()
    decision = "The TTA gain was below 0.005 calibration PQ; prefer cheaper inference."
else:
    decision = "Highest measured calibration PQ among completed first-campaign candidates."
cfg = json.loads((Path(selected["training_run"]) / "config.json").read_text())
checkpoint = Path(selected["training_run"]) / "model.pt"
selected.update(
    {
        "created_at": utc_now(),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "annotation_sha256": cfg["annotation_sha256"],
        "manifest_sha256": cfg["manifest_sha256"],
        "source_hashes": source_hashes(),
        "classical_calibration_pq": 0.08265234788493107,
        "decision": decision,
        "candidates": candidates,
        "holdout_used_for_selection": False,
    }
)
path = Path("configs/selected.json")
with path.open("x") as f:
    json.dump(selected, f, indent=2)
(Path(selected["run"]) / "postprocess.json").write_text(
    json.dumps(selected["postprocess"], indent=2)
)
print(json.dumps(selected, indent=2))
