"""Calibration-only screening; no competition test or original holdout evaluation."""

import argparse
import gc
import json
from pathlib import Path

import pandas as pd
import torch

from research.predict import ResearchPredictor, predict_cache
from solarseg.data import CompetitionData, sha256
from solarseg.engine import calibrate, evaluate_cached
from solarseg.metrics import aggregate
from solarseg.provenance import utc_now


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--run", required=True)
    p.add_argument("--data", default="data/raw")
    p.add_argument("--manifest", default="configs/manifests/train_manifest.csv")
    p.add_argument("--device")
    p.add_argument("--tta", action="store_true")
    p.add_argument("--tile", type=int, default=0)
    p.add_argument("--size", type=int)
    p.add_argument(
        "--fast-grid", action="store_true", help="Reuse components; same ordered 25-setting grid"
    )
    a = p.parse_args()
    out = Path(a.run)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "calibration-summary.json").exists():
        raise FileExistsError("Completed calibration run is immutable")
    data = CompetitionData(a.data)
    frame = pd.read_csv(a.manifest)
    stems = frame.loc[frame.role == "calibration", "stem"].tolist()
    if not stems:
        raise ValueError("Empty calibration partition")
    predictor = ResearchPredictor(a.checkpoint, a.device, a.tta, a.tile, a.size)
    cache = out / "probabilities-calibration"
    predict_cache(data, stems, predictor, cache)
    del predictor
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if a.fast_grid:
        from research.fast_calibration import calibrate as fast_calibrate

        best, rows = fast_calibrate(data, stems, cache, out)
    else:
        best = calibrate(data, stems, cache, out)
        rows = evaluate_cached(data, stems, cache, best["params"])
    record = {
        "created_at": utc_now(),
        "checkpoint_sha256": sha256(a.checkpoint),
        "manifest_sha256": sha256(a.manifest),
        "input_metadata_sha256": sha256(cache / "metadata.json"),
        "selected": best,
        "confirmed": aggregate(rows),
        "rows": rows,
        "scope": "Exploratory calibration; maximum is selection-biased; no holdout or test access",
        "grid_implementation": "component reuse" if a.fast_grid else "original sequential",
    }
    (out / "calibration-summary.json").write_text(json.dumps(record, indent=2) + "\n")
    print("SELECTED", json.dumps(best), flush=True)


if __name__ == "__main__":
    main()
