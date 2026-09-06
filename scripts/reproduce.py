"""Download the released checkpoint by hash and reproduce all test predictions."""

import argparse
import json
import time
import urllib.request
from pathlib import Path

from solarseg.data import CompetitionData, make_manifest, sha256
from solarseg.engine import Predictor, predict_cache
from solarseg.provenance import environment, source_hashes, utc_now
from solarseg.submission import write_submission

p = argparse.ArgumentParser()
p.add_argument("--data", default="data/raw")
p.add_argument("--output", default="artifacts/reproduced")
p.add_argument("--device", default="cpu")
a = p.parse_args()
selected = json.loads(Path("configs/selected.json").read_text())
out = Path(a.output)
out.mkdir(parents=True, exist_ok=True)
checkpoint = out / "model.pt"
if not checkpoint.exists():
    urllib.request.urlretrieve(
        "https://github.com/srivatsav-kannan/solar-seg/releases/download/baseline-v0.1/model.pt",
        checkpoint,
    )
if sha256(checkpoint) != selected["checkpoint_sha256"]:
    raise ValueError("Downloaded checkpoint failed checksum validation")
data = CompetitionData(a.data)
audit = make_manifest(data, out / "manifests")
if (
    audit["split_manifest_sha256"] != selected["manifest_sha256"]
    or audit["train_annotation_sha256"] != selected["annotation_sha256"]
):
    raise ValueError("Official input/split fingerprint mismatch")
predictor = Predictor(checkpoint, device=a.device, tta=selected["tta"], tile=selected["tile"])
started = time.monotonic()
stems = sorted(data.test_paths)
predict_cache(data, stems, predictor, out / "probabilities-test")
validation = write_submission(
    stems, out / "probabilities-test", selected["postprocess"], out / "submission.csv"
)
result = {
    "created_at": utc_now(),
    "validation": validation,
    "seconds": time.monotonic() - started,
    "environment": environment(),
    "source_hashes": source_hashes(),
    "device": a.device,
}
(out / "reproduction.json").write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
