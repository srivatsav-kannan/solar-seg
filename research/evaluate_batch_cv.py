"""Freeze all three paired comparisons before evaluating any outer observation."""

import gc
import json
from pathlib import Path

import pandas as pd
import torch

from research.comparison import paired_comparison
from research.predict import ResearchPredictor, predict_cache
from solarseg.data import CompetitionData, sha256
from solarseg.engine import evaluate_cached
from solarseg.metrics import aggregate
from solarseg.provenance import source_hashes, utc_now


def read(path):
    return json.loads(Path(path).read_text())


def main():
    plan_path = Path("configs/batch-dice-cv-02.json")
    plan = read(plan_path)
    index_path = Path("configs/scoring-cv-02/index.json")
    assert sha256(index_path) == plan["index_sha256"]
    index = read(index_path)
    data = CompetitionData("data/raw")
    out = Path("artifacts/cv-02/paired-batch-dice")
    out.mkdir(parents=True, exist_ok=True)
    if (out / "comparison.json").exists():
        raise FileExistsError("Completed outer comparison is immutable")
    fits = []
    groups = {}
    for entry in plan["folds"]:
        fold = entry["fold"]
        manifest = Path(entry["manifest"])
        assert sha256(manifest) == entry["manifest_sha256"] == index["records"][str(fold)]["sha256"]
        frame = pd.read_csv(manifest)
        groups.update(frame.set_index("stem").group.to_dict())
        train_stems = frame.loc[frame.role == "train", "stem"].tolist()
        outer = frame.loc[frame.role == "holdout", "stem"].tolist()
        assert set(outer).isdisjoint(train_stems)
        assert set(outer).isdisjoint(frame.loc[frame.role == "calibration", "stem"])
        candidate_root = (
            Path("artifacts/cv-02")
            if fold == 4
            else Path("artifacts/cloud-batch-cv-02/batch-dice-cv-02")
        )
        checkpoints = {
            "baseline": Path(entry["parent_checkpoint"]),
            "candidate": candidate_root / f"batch-dice-fold-{fold}/model.pt",
        }
        for kind, checkpoint in checkpoints.items():
            cfg = read(checkpoint.parent / "config.json")
            assert cfg["train_stems"] == train_stems
            assert cfg["manifest_sha256"] == entry["manifest_sha256"]
            assert cfg["annotation_sha256"] == sha256(data.annotation_path)
            assert sha256(checkpoint) == cfg["checkpoint_sha256"]
            if kind == "candidate":
                assert all(cfg[k] == v for k, v in plan["recipe"].items())
                assert cfg["initial_checkpoint_sha256"] == entry["parent_checkpoint_sha256"]
                assert cfg["device"] == entry["device"]
            else:
                assert sha256(checkpoint) == entry["parent_checkpoint_sha256"]
            prefix = "baseline" if kind == "baseline" else "batch-dice"
            calibration_path = Path(
                f"artifacts/cv-02/{prefix}-full-fold-{fold}/calibration-summary.json"
            )
            calibration = read(calibration_path)
            assert calibration["manifest_sha256"] == entry["manifest_sha256"]
            assert calibration["checkpoint_sha256"] == sha256(checkpoint)
            metadata = read(calibration_path.parent / "probabilities-calibration/metadata.json")
            assert metadata["tta"] and metadata["tile"] == 0
            assert metadata["checkpoint_sha256"] == sha256(checkpoint)
            assert calibration["input_metadata_sha256"] == sha256(
                calibration_path.parent / "probabilities-calibration/metadata.json"
            )
            calibration_stems = frame.loc[frame.role == "calibration", "stem"].tolist()
            assert metadata["input_sha256"] == {
                stem: sha256(data.image_path(stem)) for stem in calibration_stems
            }
            assert {row["stem"] for row in calibration["rows"]} == set(calibration_stems)
            fits.append(
                {
                    "fold": fold,
                    "kind": kind,
                    "checkpoint": str(checkpoint),
                    "checkpoint_sha256": sha256(checkpoint),
                    "manifest_sha256": sha256(manifest),
                    "calibration_sha256": sha256(calibration_path),
                    "params": calibration["selected"]["params"],
                    "outer_stems": outer,
                }
            )
    frozen_path = out / "frozen.json"
    frozen = {
        "plan_sha256": sha256(plan_path),
        "fits": fits,
        "inference": plan["inference"],
        "inference_device": "mps",
        "promotion": plan["promotion"],
        "evaluation_source_sha256": sha256(__file__),
        "core_source_hashes": source_hashes(),
        "research_source_hashes": {
            name: sha256(name)
            for name in ["research/predict.py", "research/models.py", "research/comparison.py"]
        },
    }
    if frozen_path.exists():
        previous = read(frozen_path)
        assert {k: v for k, v in previous.items() if k != "created_at"} == frozen
    else:
        with frozen_path.open("x") as stream:
            json.dump({"created_at": utc_now(), **frozen}, stream, indent=2)
    # No outer prediction or score is read until every fit and threshold is frozen.
    all_rows = {"baseline": [], "candidate": []}
    per_fold = {}
    for fit in fits:
        assert sha256(fit["checkpoint"]) == fit["checkpoint_sha256"]
        assert source_hashes() == frozen["core_source_hashes"]
        assert all(
            sha256(name) == digest for name, digest in frozen["research_source_hashes"].items()
        )
        run = out / f"{fit['kind']}-fold-{fit['fold']}"
        run.mkdir(exist_ok=True)
        result_path = run / "outer.json"
        if result_path.exists():
            result = read(result_path)
            assert result["frozen_sha256"] == sha256(frozen_path)
        else:
            predictor = ResearchPredictor(fit["checkpoint"], device="mps", tta=True, tile=0)
            cache = run / "probabilities-outer"
            predict_cache(data, fit["outer_stems"], predictor, cache)
            del predictor
            gc.collect()
            torch.mps.empty_cache()
            rows = evaluate_cached(data, fit["outer_stems"], cache, fit["params"])
            result = {
                "created_at": utc_now(),
                "frozen_sha256": sha256(frozen_path),
                "params": fit["params"],
                "rows": rows,
                **aggregate(rows),
            }
            result_path.write_text(json.dumps(result, indent=2) + "\n")
        all_rows[fit["kind"]].extend(result["rows"])
        per_fold.setdefault(str(fit["fold"]), {})[fit["kind"]] = result["pq"]
        print("OUTER", fit["fold"], fit["kind"], result["pq"], flush=True)
    paired = paired_comparison(all_rows["baseline"], all_rows["candidate"], groups)
    fold_gains = {
        fold: values["candidate"] - values["baseline"] for fold, values in per_fold.items()
    }
    gates = {
        "pooled_gain_at_least_0.01": paired["pq_gain"] >= 0.01,
        "no_fold_loss_worse_than_0.01": min(fold_gains.values()) >= -0.01,
        "positive_paired_time_group_lower_bound": paired["paired_block_bootstrap_95ci"][0] > 0,
    }
    result = {
        "created_at": utc_now(),
        "frozen_sha256": sha256(frozen_path),
        **paired,
        "fold_pq": per_fold,
        "fold_gains": fold_gains,
        "statistical_gates": gates,
        "statistical_gates_passed": all(gates.values()),
        "morphology_review": "pending",
        "scope": "Subsequent development on the same corpus; shared inner calibration and earlier exposure limit independence. New recipe combines batch Dice and extra training. No competition test inference or submission.",
    }
    (out / "comparison.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {k: v for k, v in result.items() if k not in ["reference", "candidate"]}, indent=2
        )
    )


if __name__ == "__main__":
    main()
