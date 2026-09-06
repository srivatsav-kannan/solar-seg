"""Command-line entry points: python -m solarseg --help."""

import argparse
import json
from pathlib import Path

import pandas as pd

from solarseg.data import CompetitionData, make_manifest
from solarseg.engine import Predictor, calibrate, evaluate_and_save, predict_cache, train
from solarseg.submission import write_submission


def main():
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=["audit", "train", "calibrate", "evaluate", "predict"])
    p.add_argument("--data", default="data/raw")
    p.add_argument("--manifests", default="artifacts/manifests")
    p.add_argument("--run", default="artifacts/classical-v1")
    p.add_argument("--checkpoint")
    p.add_argument("--steps", type=int, default=1600)
    p.add_argument("--size", type=int, default=1024)
    p.add_argument("--crop", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=6)
    p.add_argument("--width", type=int, default=16)
    p.add_argument("--device")
    p.add_argument("--tta", action="store_true")
    p.add_argument("--tile", type=int, default=0)
    a = p.parse_args()
    run = Path(a.run)
    run.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(a.manifests) / "train_manifest.csv"
    data = CompetitionData(a.data)
    if a.command == "audit":
        print(json.dumps(make_manifest(data, a.manifests), indent=2))
        return
    if a.command == "train":
        train(
            a.data,
            manifest_path,
            run,
            a.steps,
            a.size,
            a.crop,
            a.batch_size,
            a.width,
            device=a.device,
        )
        return
    manifest = pd.read_csv(manifest_path)
    role = (
        "calibration"
        if a.command == "calibrate"
        else "holdout"
        if a.command == "evaluate"
        else "test"
    )
    stems = (
        sorted(data.test_paths)
        if role == "test"
        else manifest.loc[manifest.role == role, "stem"].tolist()
    )
    predictor = Predictor(a.checkpoint, a.size, a.device, a.tta, a.tile)
    cache = run / f"probabilities-{role}"
    predict_cache(data, stems, predictor, cache)
    if a.command == "calibrate":
        print(calibrate(data, stems, cache, run))
    else:
        params = json.loads((run / "postprocess.json").read_text())
        if a.command == "evaluate":
            if (run / "holdout.json").exists():
                raise FileExistsError("Locked holdout was already evaluated for this run")
            print(evaluate_and_save(data, stems, cache, params, run / "holdout.json"))
        else:
            print(write_submission(stems, cache, params, run / "submission.csv"))


if __name__ == "__main__":
    main()
