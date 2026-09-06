"""Submit the all-data batch-Dice model only after paired validation and fresh replay."""

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


def build_gate():
    selection_path = Path("configs/batch-dice-full-selected.json")
    selected = read(selection_path)
    plan_path = Path("configs/batch-dice-full-fit-02.json")
    plan = read(plan_path)
    run = Path(selected["run"])
    cfg = read(run / "config.json")
    cv_root = Path("artifacts/cv-02/paired-batch-dice")
    cv = read(cv_root / "comparison.json")
    frozen = read(cv_root / "frozen.json")
    morphology = read(cv_root / "morphology-review.json")
    replay = read("artifacts/batch-replay/record.json")
    quality = read("reports/quality-checks.json")
    parity = read("reports/fast-calibration-parity.json")
    data = CompetitionData("data/raw")
    manifest = pd.read_csv(plan["manifest"])
    checkpoint_sha = sha256(selected["checkpoint"])
    embedded = torch.load(selected["checkpoint"], map_location="cpu", weights_only=True)["config"]
    validation = validate_submission(run / "submission.csv", data.test_paths)
    inference = read(run / "probabilities-test/metadata.json")
    source = source_hashes()
    consumed_training = ["research/train.py", "research/models.py", "research/losses.py"]
    checks = {
        "paired_outer_validation": (
            cv["statistical_gates_passed"]
            and cv["pq_gain"] >= 0.01
            and min(cv["fold_gains"].values()) >= -0.01
            and cv["paired_block_bootstrap_95ci"][0] > 0
            and cv["frozen_sha256"] == sha256(cv_root / "frozen.json")
            and frozen["plan_sha256"] == sha256("configs/batch-dice-cv-02.json")
            and selected["cv_comparison_sha256"] == sha256(cv_root / "comparison.json")
            and frozen["core_source_hashes"] == source
            and all(sha256(fit["checkpoint"]) == fit["checkpoint_sha256"] for fit in frozen["fits"])
            and all(
                sha256(name) == digest for name, digest in frozen["research_source_hashes"].items()
            )
        ),
        "morphology": morphology["passed"]
        and morphology["comparison_sha256"] == sha256(cv_root / "comparison.json"),
        "frozen_full_training_recipe": (
            sha256(plan_path) == selected["full_fit_plan_sha256"]
            and plan["created_at"] < cfg["created_at"]
            and plan["created_at"] < frozen["created_at"]
            and all(cfg[k] == v for k, v in plan["recipe"].items())
            and all(cfg[k] == v for k, v in embedded.items())
            and cfg["initial_checkpoint_sha256"]
            == plan["initialize_sha256"]
            == sha256(plan["initialize"])
            and cfg["source_hashes"] == plan["source_hashes"] == source
            and all(
                cfg["research_source_hashes"][name]
                == plan["training_source_hashes"][name]
                == sha256(name)
                for name in consumed_training
            )
            and read(run / "history.json")[-1]["step"] == 3000
            and cfg["checkpoint_sha256"] == checkpoint_sha == selected["checkpoint_sha256"]
            and all(selected[k] == v for k, v in plan["inference"].items())
            and read(run / "postprocess.json") == selected["postprocess"]
        ),
        "approved_inputs": (
            len(manifest) == 707
            and manifest.stem.is_unique
            and set(manifest.stem) == set(data.by_stem)
            and set(manifest.role) == {"train"}
            and cfg["train_stems"] == manifest.stem.tolist()
            and cfg["manifest_sha256"] == sha256(plan["manifest"]) == plan["manifest_sha256"]
            and cfg["annotation_sha256"] == sha256(data.annotation_path)
            and all(
                sha256(data.image_path(row.stem)) == row.sha256 for row in manifest.itertuples()
            )
            and inference["input_sha256"] == {s: sha256(p) for s, p in data.test_paths.items()}
        ),
        "metric_and_grid_correctness": (
            quality["passed"]
            and quality["organizer_parity_passed"]
            and quality["source_hashes"] == source
            and parity["passed"]
            and parity["settings"] == 25
            and parity["all_results_identical"]
            and parity["all_selected_rows_identical"]
        ),
        "fresh_cpu_reproduction": (
            validation["valid"]
            and validation["images_processed"] == 180
            and replay["passed"]
            and replay["cache_images_before"] == 0
            and replay["selection_sha256"] == sha256(selection_path)
            and replay["notebook_sha256"] == sha256("notebooks/batch_dice.ipynb")
            and replay["submission_sha256"] == validation["sha256"]
            and replay["source_hashes"] == replay["proof"]["source_hashes"] == source
            and replay["proof"]["checkpoint_sha256"] == checkpoint_sha
            and replay["proof"]["cv_comparison_sha256"] == selected["cv_comparison_sha256"]
            and replay["proof"]["training_manifest_sha256"] == plan["manifest_sha256"]
            and not replay["proof"]["training_rerun"]
            and not replay["proof"]["holdout_rerun"]
            and inference["checkpoint"] == checkpoint_sha
            and inference["tta"] == selected["tta"]
            and inference["tile"] == selected["tile"]
        ),
    }
    record = {
        "created_at": utc_now(),
        "checks": checks,
        "ready_to_submit": all(checks.values()),
        "submission_sha256": validation["sha256"],
        "checkpoint_sha256": checkpoint_sha,
        "selection_sha256": sha256(selection_path),
        "cv_comparison_sha256": selected["cv_comparison_sha256"],
        "source_hashes": source,
        "validation_scope": plan["validation_scope"],
        "validation": validation,
    }
    (run / "batch-gates.json").write_text(json.dumps(record, indent=2) + "\n")
    if not record["ready_to_submit"]:
        raise RuntimeError(f"Submission gate failed: {checks}")
    return run, record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    run, gate = build_gate()
    from kaggle import api

    competition = "filament-segmentation-2026"
    limits = json.loads(str(api.competition_get_submission_limits(competition)))
    if limits.get("numAllowedNow", 0) < 1:
        raise RuntimeError(f"No submission quota: {limits}")
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
        "description": f"Batch Dice + extra training; paired grouped CV passed; all 707 approved images | {gate['submission_sha256'][:12]}",
        "limits_before": limits,
        "gate_sha256": sha256(run / "batch-gates.json"),
    }
    with path.open("x") as stream:
        json.dump(record, stream, indent=2)
    response = api.competition_submit(
        str(run / "submission.csv"), record["description"], competition
    )
    record.update(response=json.loads(str(response)), state="response_received_check_server_status")
    path.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
