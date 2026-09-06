"""Submit a new all-data fit of the unchanged, already-validated recipe."""

import argparse
import json
from pathlib import Path

import pandas as pd
import torch

from solarseg.data import CompetitionData, sha256
from solarseg.provenance import source_hashes, utc_now
from solarseg.submission import validate_submission


def read(path):
    return json.loads(Path(path).read_text())


def build_refit_gate(data_root="data/raw"):
    plan_path = Path("configs/full-refit-v1.json")
    selection_path = Path("configs/full-refit-selected.json")
    plan, selection = read(plan_path), read(selection_path)
    parent_path = Path("configs/selected.json")
    parent = read(parent_path)
    run = Path(selection["run"])
    cfg = read(run / "config.json")
    parent_cfg = read(Path(parent["training_run"]) / "config.json")
    provenance = read(run / "full-refit-provenance.json")
    history = read(run / "history.json")
    parent_gate = read(Path(parent["run"]) / "gates.json")
    quality = read("reports/quality-checks.json")
    replay = read("artifacts/refit-replay/record.json")
    proof = replay["proof"]
    manifest = pd.read_csv(plan["manifest"])
    data = CompetitionData(data_root)
    inference = read(run / "probabilities-test/metadata.json")
    validation = validate_submission(run / "submission.csv", data.test_paths)
    source = source_hashes()
    embedded = torch.load(selection["checkpoint"], map_location="cpu", weights_only=True)["config"]
    checkpoint_sha = sha256(selection["checkpoint"])
    test_hashes = {stem: sha256(path) for stem, path in sorted(data.test_paths.items())}
    checks = {
        "approved_training_inputs": (
            len(manifest) == 707
            and manifest.stem.is_unique
            and set(manifest.role) == {"train"}
            and set(manifest.stem) == set(data.by_stem)
            and cfg["train_stems"] == manifest.stem.tolist()
            and all(sha256(data.image_path(r.stem)) == r.sha256 for r in manifest.itertuples())
            and sha256(data.annotation_path) == plan["recipe"]["annotation_sha256"]
            and sha256(plan["manifest"]) == plan["manifest_sha256"] == cfg["manifest_sha256"]
        ),
        "unchanged_validated_recipe": (
            sha256(parent_path)
            == plan["parent_selection_sha256"]
            == selection["parent_selection_sha256"]
            and parent_gate["ready_to_submit"]
            and parent_gate["selection_sha256"] == sha256(parent_path)
            and parent_gate["source_hashes"]
            == source
            == plan["source_hashes"]
            == cfg["source_hashes"]
            and all(cfg[k] == v == parent_cfg[k] for k, v in plan["recipe"].items())
            and all(selection[k] == v == parent[k] for k, v in plan["inference"].items())
            and read(run / "postprocess.json") == parent["postprocess"]
            and sha256(parent["checkpoint"]) == plan["parent_checkpoint_sha256"]
            and plan["created_at"] < cfg["created_at"]
        ),
        "completed_new_fit": (
            checkpoint_sha
            == selection["checkpoint_sha256"]
            == cfg["checkpoint_sha256"]
            == provenance["checkpoint_sha256"]
            and sha256(plan_path) == provenance["plan_sha256"] == selection["plan_sha256"]
            and all(cfg[k] == v for k, v in embedded.items())
            and history[-1]["step"] == plan["recipe"]["steps"]
            and history[-1]["seconds"] > 0
        ),
        "correctness_and_input_identity": (
            quality["passed"]
            and quality["organizer_parity_passed"]
            and quality["source_hashes"] == source
            and inference["input_sha256"] == test_hashes
            and inference["checkpoint"] == checkpoint_sha
            and inference["tta"] == selection["tta"]
            and inference["tile"] == selection["tile"]
        ),
        "fresh_independent_notebook_replay": (
            validation["valid"]
            and validation["images_processed"] == len(data.test_paths) == 180
            and replay["passed"]
            and replay["cache_images_before"] == 0
            and replay["selection_sha256"] == sha256(selection_path)
            and replay["notebook_sha256"] == sha256("notebooks/full_refit.ipynb")
            and replay["source_hashes"] == proof["source_hashes"] == source
            and replay["submission_sha256"] == validation["sha256"] == proof["validation"]["sha256"]
            and proof["completed"]
            and proof["full_data_refit"]
            and not proof["holdout_rerun"]
            and proof["checkpoint_sha256"] == checkpoint_sha
            and proof["training_manifest_sha256"] == plan["manifest_sha256"]
            and proof["parent_checkpoint_sha256"] == parent["checkpoint_sha256"]
        ),
    }
    result = {
        "created_at": utc_now(),
        "checks": checks,
        "ready_to_submit": all(checks.values()),
        "submission_sha256": validation["sha256"],
        "checkpoint_sha256": checkpoint_sha,
        "selection_sha256": sha256(selection_path),
        "source_hashes": source,
        "validation_scope": plan["validation_scope"],
        "validation": validation,
        "parent_gate_sha256": sha256(Path(parent["run"]) / "gates.json"),
        "scope": "All-data final fitting of the previously validated recipe; new weights have no independent internal holdout. Final-entry paperwork is separate.",
    }
    (run / "refit-gates.json").write_text(json.dumps(result, indent=2) + "\n")
    if not result["ready_to_submit"]:
        raise RuntimeError(f"Refit gate failure: {checks}")
    return run, result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--check-only", action="store_true")
    p.add_argument("--data", default="data/raw")
    args = p.parse_args()
    run, gate = build_refit_gate(args.data)
    from kaggle import api

    competition = "filament-segmentation-2026"
    limits = json.loads(str(api.competition_get_submission_limits(competition)))
    if limits.get("numAllowedNow", 0) < 1:
        raise RuntimeError(f"No submissions available: {limits}")
    print(json.dumps({"checks": gate["checks"], "limits": limits}, indent=2))
    if args.check_only:
        return
    path = run / "kaggle-submission.json"
    record = {
        "created_at": utc_now(),
        "state": "intent_recorded",
        "competition": competition,
        "submission_sha256": gate["submission_sha256"],
        "checkpoint_sha256": gate["checkpoint_sha256"],
        "description": f"Unchanged validated U-Net recipe, all 707 approved training images | {gate['submission_sha256'][:12]}",
        "limits_before": limits,
        "gate_sha256": sha256(run / "refit-gates.json"),
    }
    # Exclusive creation prevents blind retries after an uncertain upload.
    with path.open("x") as f:
        json.dump(record, f, indent=2)
    response = api.competition_submit(
        str(run / "submission.csv"), record["description"], competition
    )
    record.update(response=json.loads(str(response)), state="response_received_check_server_status")
    path.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
