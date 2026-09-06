"""Freeze a bounded nested holdout comparison before campaign-02 model selection."""

import json
from pathlib import Path

import pandas as pd

from scripts.make_nested_folds import assign_roles
from solarseg.data import sha256
from solarseg.provenance import utc_now


def build_scoring_folds(frame):
    frame = frame.reset_index(drop=True)
    if not frame.stem.is_unique:
        raise ValueError("Repeated physical observation")
    for key in ["group", "pixel_sha256"]:
        if frame.groupby(key).fold.nunique().max() != 1:
            raise ValueError(f"Original folds divide {key}")
    results = {}
    for outer in [2, 3, 4]:
        frame_outer = assign_roles(frame, frame.fold == outer, frame.fold == 1)
        # Outer-neighbor observations must not appear in inner calibration either.
        outer_only = assign_roles(frame, frame.fold == outer, frame.fold == -1)
        frame_outer.loc[
            (frame_outer.role == "calibration") & (outer_only.role == "embargo"), "role"
        ] = "embargo"
        frame_outer["outer_fold"] = outer
        results[outer] = frame_outer
    return results


def main():
    source = Path("configs/manifests/train_manifest.csv")
    out = Path("configs/scoring-cv-02")
    if out.exists():
        raise FileExistsError("Frozen scoring folds are immutable")
    out.mkdir()
    records = {}
    for outer, frame in build_scoring_folds(pd.read_csv(source)).items():
        path = out / f"fold-{outer}.csv"
        frame.to_csv(path, index=False)
        records[str(outer)] = {
            "path": str(path),
            "sha256": sha256(path),
            "roles": frame.role.value_counts().to_dict(),
        }
    plan = {
        "created_at": utc_now(),
        "source_manifest_sha256": sha256(source),
        "generator_sha256": sha256(__file__),
        "outer_folds": [2, 3, 4],
        "inner_calibration": "Original fold 1, minus any outer-holdout embargo neighbors",
        "embargo_days": 3,
        "records": records,
        "protocol": "Fit each model from scratch on train only. Tune the identical bounded reconstruction grid on that fit's calibration images. Freeze the candidate before outer inference. Pool annotator counts across the three outer folds; keep physical/time groups together for paired bootstrap.",
        "promotion": "Require pooled PQ gain >=0.01 over identically split baseline, no outer-fold loss worse than 0.01, and positive paired temporal-group-bootstrap lower bound where supported. Review morphology. Do not tune on the outer scores.",
        "final_training": "After recipe selection, refit on all approved training observations with fixed hyperparameters. Record that this checkpoint has recipe-level CV evidence, not the old protected-holdout score.",
        "reason_for_bounded_design": "Prioritize scored model development within available GPU time. Three outer folds with one fixed inner calibration partition replace the more expensive five-by-three plan for this campaign; these are not results from that original full nested-CV plan.",
        "limitations": "Subsequent development on a previously analyzed corpus. The architecture was informed by previous experiments. This is not a previously untouched external test set. Shared inner calibration limits selection independence; future full nested and chronological/site stress checks remain desirable.",
        "status": "Frozen before campaign-02 selection and before any of these CV fits",
    }
    (out / "index.json").write_text(json.dumps(plan, indent=2) + "\n")
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
