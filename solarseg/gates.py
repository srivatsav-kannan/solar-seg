"""Fail-closed release evidence bound to one selected model and CSV."""

import json
from pathlib import Path

import pandas as pd

from solarseg.data import CompetitionData, sha256
from solarseg.provenance import source_hashes, utc_now
from solarseg.submission import validate_submission


def read_json(path):
    return json.loads(Path(path).read_text())


def build_gates(run, selected_path="configs/selected.json", data_root="data/raw"):
    run = Path(run)
    selected = read_json(selected_path)
    quality = read_json("reports/quality-checks.json")
    holdout = read_json(run / "holdout.json")
    morphology = read_json(run / "morphology-review.json")
    replay = read_json("artifacts/replay/record.json")
    audit = read_json("artifacts/manifests/audit.json")
    params = read_json(run / "postprocess.json")
    data = CompetitionData(data_root)
    validation = validate_submission(run / "submission.csv", data.test_paths)
    current_source = source_hashes()
    integrity = True
    for split in ["train", "test"]:
        manifest = pd.read_csv(f"artifacts/manifests/{split}_manifest.csv")
        integrity &= all(sha256(data.image_path(r.stem)) == r.sha256 for r in manifest.itertuples())
    checks = {
        "G0_inputs": integrity
        and sha256(data.annotation_path) == audit["train_annotation_sha256"]
        and sha256("artifacts/manifests/train_manifest.csv") == selected["manifest_sha256"],
        "G1_correctness": quality["passed"]
        and quality["organizer_parity_passed"]
        and quality["source_hashes"] == current_source,
        "G2_calibration": selected["calibration_pq"] > selected["classical_calibration_pq"]
        and params == selected["postprocess"]
        and sha256(selected["checkpoint"]) == selected["checkpoint_sha256"],
        "G3_holdout_morphology": holdout["pq"] >= 0.15
        and holdout["images"] == audit["roles"]["holdout"]
        and holdout["params"] == params
        and selected["created_at"] < holdout["created_at"]
        and morphology["passed"]
        and morphology["selection_sha256"] == sha256(selected_path),
        "G4_reproduction": replay["passed"]
        and replay["source_hashes"] == current_source
        and replay["submission_sha256"] == validation["sha256"]
        and replay["notebook_sha256"] == sha256("notebooks/canonical.ipynb")
        and validation["images_processed"] == audit["test"],
    }
    result = {
        "created_at": utc_now(),
        "checks": checks,
        "ready_to_submit": all(checks.values()),
        "submission_sha256": validation["sha256"],
        "selection_sha256": sha256(selected_path),
        "source_hashes": current_source,
        "holdout_pq": holdout["pq"],
        "scope": "One baseline CSV submission; G6 final-entry requirements remain separate",
    }
    (run / "gates.json").write_text(json.dumps(result, indent=2))
    if not result["ready_to_submit"]:
        raise RuntimeError(f"Submission gate failure: {checks}")
    return result
