"""Refit the previously validated, unchanged baseline recipe on all approved labels."""

import json
from pathlib import Path

import pandas as pd

from solarseg.data import CompetitionData, sha256
from solarseg.engine import train
from solarseg.provenance import source_hashes, utc_now


def main():
    plan_path = Path("configs/full-refit-v1.json")
    plan = json.loads(plan_path.read_text())
    data = CompetitionData("data/raw")
    manifest = pd.read_csv(plan["manifest"])
    assert sha256(plan["manifest"]) == plan["manifest_sha256"]
    assert sha256("configs/selected.json") == plan["parent_selection_sha256"]
    assert sha256("artifacts/unet-v2/model.pt") == plan["parent_checkpoint_sha256"]
    assert source_hashes() == plan["source_hashes"]
    assert set(manifest.role) == {"train"} and manifest.stem.is_unique
    assert set(manifest.stem) == set(data.by_stem) and len(manifest) == 707
    assert sha256(data.annotation_path) == plan["recipe"]["annotation_sha256"]
    assert all(sha256(data.image_path(row.stem)) == row.sha256 for row in manifest.itertuples())
    recipe = plan["recipe"]
    fields = ["steps", "size", "crop", "batch_size", "width", "seed"]
    result = train(
        "data/raw", plan["manifest"], plan["run"], device="mps", **{k: recipe[k] for k in fields}
    )
    assert all(result[k] == value for k, value in recipe.items())
    record = {
        "created_at": utc_now(),
        "plan_sha256": sha256(plan_path),
        "launcher_sha256": sha256(__file__),
        "checkpoint_sha256": result["checkpoint_sha256"],
        "approved_training_observations": len(manifest),
        "recipe_matches_parent": True,
        "validation_scope": plan["validation_scope"],
    }
    (Path(plan["run"]) / "full-refit-provenance.json").write_text(
        json.dumps(record, indent=2) + "\n"
    )
    print(json.dumps(record, indent=2), flush=True)


if __name__ == "__main__":
    main()
