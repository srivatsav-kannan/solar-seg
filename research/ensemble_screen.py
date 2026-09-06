"""Prespecified convex probability ensembles on the original calibration images."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from solarseg.data import CompetitionData, sha256
from solarseg.engine import calibrate, evaluate_cached
from solarseg.metrics import aggregate
from solarseg.provenance import utc_now


def native(prob):
    if prob.ndim != 2 or not np.isfinite(prob).all() or prob.min() < 0 or prob.max() > 1:
        raise ValueError("Invalid source probability map")
    return np.asarray(
        Image.fromarray(prob.astype(np.float32)).resize((2048, 2048), Image.Resampling.BILINEAR)
    )


def main():
    plan = json.loads(Path("configs/scoring-campaign-02.json").read_text())
    manifest = Path(plan["split"])
    if sha256(manifest) != plan["split_sha256"]:
        raise ValueError("Frozen split differs")
    data = CompetitionData("data/raw")
    frame = pd.read_csv(manifest)
    stems = frame.loc[frame.role == "calibration", "stem"].tolist()
    input_hashes = {s: sha256(data.image_path(s)) for s in stems}
    base = Path("artifacts") / plan["cheap_ensemble_screen"]["base"] / "probabilities-calibration"
    base_meta = json.loads((base / "metadata.json").read_text())
    if base_meta["input_sha256"] != input_hashes:
        raise ValueError("Base predictions do not cover exactly the calibration inputs")
    for candidate in plan["cheap_ensemble_screen"]["components"]:
        component = Path("artifacts") / candidate["run"] / "probabilities-calibration"
        component_meta = json.loads((component / "metadata.json").read_text())
        if component_meta["input_sha256"] != input_hashes:
            raise ValueError("Component predictions do not match calibration inputs")
        weight = candidate["weight"]
        if not 0 < weight < 1:
            raise ValueError("Ensemble must be a convex probability average")
        out = Path("artifacts") / f"blend-{candidate['run']}-{int(100 * weight)}"
        cache = out / "probabilities-calibration"
        if (out / "calibration-summary.json").exists():
            continue
        cache.mkdir(parents=True, exist_ok=True)
        fingerprint = {
            "created_at": utc_now(),
            "base": str(base),
            "component": str(component),
            "component_weight": weight,
            "base_metadata_sha256": sha256(base / "metadata.json"),
            "component_metadata_sha256": sha256(component / "metadata.json"),
            "script_sha256": sha256(__file__),
            "input_sha256": input_hashes,
            "source_probability_sha256": {
                str(p): sha256(p)
                for d in [base, component]
                for p in [d / f"{s}.npy" for s in stems]
            },
        }
        meta_path = cache / "metadata.json"
        if meta_path.exists():
            old = json.loads(meta_path.read_text())
            fingerprint["created_at"] = old["created_at"]
            if old != fingerprint:
                raise ValueError("Ensemble source/implementation changed")
        meta_path.write_text(json.dumps(fingerprint, indent=2) + "\n")
        for stem in stems:
            path = cache / f"{stem}.npy"
            if not path.exists():
                p0 = native(np.load(base / f"{stem}.npy"))
                p1 = native(np.load(component / f"{stem}.npy"))
                np.save(path, (1 - weight) * p0 + weight * p1)
        best = calibrate(data, stems, cache, out)
        rows = evaluate_cached(data, stems, cache, best["params"])
        record = {
            "created_at": utc_now(),
            "plan_sha256": sha256("configs/scoring-campaign-02.json"),
            "manifest_sha256": sha256(manifest),
            "input_metadata_sha256": sha256(meta_path),
            "selected": best,
            "confirmed": aggregate(rows),
            "rows": rows,
            "scope": "Exploratory original calibration only; no holdout or test access",
        }
        (out / "calibration-summary.json").write_text(json.dumps(record, indent=2) + "\n")
        print("SELECTED", str(out), json.dumps(best), flush=True)


if __name__ == "__main__":
    main()
