"""Run only a frozen reconstruction grid against calibration predictions."""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from research.reconstruction import reconstruct
from solarseg.data import CompetitionData, sha256
from solarseg.metrics import aggregate, evaluate_image
from solarseg.provenance import utc_now


def main():
    plan_path = Path("configs/reconstruction-screen-02.json")
    plan = json.loads(plan_path.read_text())
    data = CompetitionData("data/raw")
    manifest = Path(plan["manifest"])
    if sha256(manifest) != plan["manifest_sha256"]:
        raise ValueError("Frozen manifest changed")
    frame = pd.read_csv(manifest)
    stems = frame.loc[frame.role == "calibration", "stem"].tolist()
    cache = Path(plan["probabilities"])
    metadata = json.loads((cache / "metadata.json").read_text())
    if metadata["input_sha256"] != {s: sha256(data.image_path(s)) for s in stems}:
        raise ValueError("Calibration cache does not match inputs")
    out = Path(plan["run"])
    out.mkdir(parents=True, exist_ok=True)
    if (out / "results.json").exists():
        raise FileExistsError("Completed reconstruction screen is immutable")
    source_hashes = {
        str(p): sha256(p) for p in [Path(__file__), Path("research/reconstruction.py")]
    }
    started = time.monotonic()
    rows = [[] for _ in plan["grid"]]
    for stem in tqdm(stems, desc="Reconstruction screen"):
        prob = np.load(cache / f"{stem}.npy")
        for i, params in enumerate(plan["grid"]):
            rows[i].extend(evaluate_image(data, stem, reconstruct(prob, **params)))
    results = [dict(params=p, **aggregate(r)) for p, r in zip(plan["grid"], rows, strict=True)]
    best = max(range(len(results)), key=lambda i: results[i]["pq"])
    record = {
        "created_at": utc_now(),
        "plan_sha256": sha256(plan_path),
        "input_metadata_sha256": sha256(cache / "metadata.json"),
        "source_hashes": source_hashes,
        "seconds": time.monotonic() - started,
        "results": results,
        "selected": results[best],
        "rows": rows[best],
        "scope": "Original calibration only; no holdout/test access; selection-biased screen",
    }
    (out / "results.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({k: v for k, v in record.items() if k != "rows"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
