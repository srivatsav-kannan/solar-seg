"""Freeze subsequent-development CV using the published physical-image groups."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from solarseg.data import sha256
from solarseg.provenance import utc_now


def assign_roles(frame, holdout, calibration):
    result = frame.copy()
    held = holdout | calibration
    dates = pd.to_datetime(frame.date).dt.as_unit("ns").astype("int64").to_numpy()
    distance = np.abs(dates[:, None] - dates[held.to_numpy()][None, :]).min(axis=1)
    result["role"] = np.select(
        [holdout, calibration, distance <= 3 * 86400 * 10**9],
        ["holdout", "calibration", "embargo"],
        default="train",
    )
    if not (result.role == "train").any():
        raise ValueError("Embargo removed every optimization observation")
    return result


def build_folds(frame, inner_splits=3, seed=2027):
    frame = frame.reset_index(drop=True)
    if not frame.stem.is_unique:
        raise ValueError("Each physical observation must occur once")
    for key in ["group", "pixel_sha256"]:
        if frame.groupby(key).fold.nunique().max() != 1:
            raise ValueError(f"Outer folds divide {key}")
    results = {}
    for outer in sorted(frame.fold.unique()):
        holdout = frame.fold == outer
        refit = assign_roles(frame, holdout, pd.Series(False, index=frame.index))
        # Inner calibration must also stay outside the outer holdout's embargo.
        inner_pool = frame.loc[refit.role == "train"]
        splitter = GroupKFold(n_splits=inner_splits, shuffle=True, random_state=seed + int(outer))
        for inner, (_, indices) in enumerate(splitter.split(inner_pool, groups=inner_pool.group)):
            calibration = frame.index.isin(inner_pool.iloc[indices].index)
            calibration = pd.Series(calibration, index=frame.index)
            result = assign_roles(frame, holdout, calibration)
            result["outer_fold"], result["inner_fold"] = int(outer), inner
            results[f"outer-{outer}/inner-{inner}.csv"] = result
        result = refit
        result["outer_fold"], result["inner_fold"] = int(outer), -1
        results[f"outer-{outer}/refit.csv"] = result
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="configs/manifests/train_manifest.csv")
    parser.add_argument("--output", default="artifacts/nested-cv")
    parser.add_argument("--index", default="configs/nested-cv.json")
    args = parser.parse_args()
    index_path, out = Path(args.index), Path(args.output)
    if index_path.exists() or out.exists():
        raise FileExistsError("Use new paths; frozen CV manifests are immutable")
    folds = build_folds(pd.read_csv(args.manifest))
    records = {}
    for name, frame in folds.items():
        path = out / name
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False)
        records[name] = {"sha256": sha256(path), "roles": frame.role.value_counts().to_dict()}
    index = {
        "created_at": utc_now(),
        "source_manifest": args.manifest,
        "source_manifest_sha256": sha256(args.manifest),
        "generator_sha256": sha256(__file__),
        "outer_folds": 5,
        "inner_folds": 3,
        "inner_seed": 2027,
        "embargo_days": 3,
        "output_directory": args.output,
        "records": records,
        "status": "Prepared before any nested-CV training; no nested-CV model results yet",
        "limitation": "Subsequent development on a previously analyzed corpus, not an untouched independent test",
    }
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text(json.dumps(index, indent=2) + "\n")
    print(f"Created {len(folds)} manifests; index: {index_path}")


if __name__ == "__main__":
    main()
