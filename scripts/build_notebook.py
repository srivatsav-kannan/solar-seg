"""Generate the canonical, executable workflow; keep model code in modules."""

import argparse
import json
from pathlib import Path

import nbformat as nbf

from solarseg.data import sha256
from solarseg.provenance import source_hashes

p = argparse.ArgumentParser()
p.add_argument("--source-commit", required=True)
p.add_argument("--release-tag", default="baseline-v0.1")
a = p.parse_args()
root = Path(__file__).resolve().parents[1]
selected = json.loads((root / "configs/selected.json").read_text())
training = json.loads((root / selected["training_run"] / "config.json").read_text())
cells = []


def md(text):
    cells.append(nbf.v4.new_markdown_cell(text.strip()))


def code(text):
    cells.append(nbf.v4.new_code_cell(text.strip()))


md("""
# Solar filaments: a reproducible instance-segmentation baseline

Predict each individual dark solar filament in a 2048 × 2048 H-alpha observation.
The ranking metric is **pooled per-annotator Panoptic Quality**, with strict IoU > 0.5.
This notebook is the canonical workflow for [solar-seg](https://github.com/srivatsav-kannan/solar-seg).
The modules contain the implementation; the notebook explains and runs the steps.

**Default run:** audit the official inputs, load the released checkpoint, inspect a calibration
example, run correctness checks, infer all test images on CPU, and validate the output CSV.
Training and holdout evaluation are explicit opt-ins because they are expensive and the
holdout has already been exposed during the recorded baseline campaign. A default inference
replay is not a claim that training has been rerun. No Kaggle submission occurs automatically.

Use only the competition's training JSON. The full public MAGFiLO labels overlap the test set.
No external images or pretrained filament models are used. Data retain their original terms.
See the repository's requirements register, literature review, validation protocol, and ledger.
""")
bootstrap = r"""
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import urllib.request
import zipfile

SOURCE_COMMIT = SOURCEVALUE
EXPECTED_SOURCE = HASHVALUE
RUN_TRAINING = False
RUN_HOLDOUT = False
INSTALL_DEPENDENCIES = Path("/kaggle").exists()
REPO = "srivatsav-kannan/solar-seg"

ROOT = Path.cwd()
if not (ROOT / "solarseg").is_dir():
    if (ROOT.parent / "solarseg").is_dir():
        ROOT = ROOT.parent
    else:
        url = f"https://github.com/{REPO}/archive/{SOURCE_COMMIT}.zip"
        payload = urllib.request.urlopen(url, timeout=120).read()
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            for member in archive.infolist():
                target = (ROOT / member.filename).resolve()
                if not target.is_relative_to(ROOT.resolve()):
                    raise ValueError("Unsafe source archive path")
            archive.extractall(ROOT)
        ROOT = ROOT / f"solar-seg-{SOURCE_COMMIT}"
for name, expected in EXPECTED_SOURCE.items():
    actual = hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
    assert actual == expected, f"Source fingerprint differs: {name}"
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
assert sys.version_info >= (3, 12), "Use Python 3.12 or newer with the pinned dependencies"
if INSTALL_DEPENDENCIES:
    subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", "-r", "requirements.txt"], check=True)
print("Source revision:", SOURCE_COMMIT)
"""
code(
    bootstrap.replace("SOURCEVALUE", repr(a.source_commit)).replace(
        "HASHVALUE", repr(source_hashes())
    )
)
code(f"""
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from solarseg.data import CompetitionData, make_manifest, read_image, sha256
from solarseg.engine import Predictor, train, predict_cache, calibrate, evaluate_and_save
from solarseg.metrics import aggregate, counts
from solarseg.postprocess import encode, decode, instances
from solarseg.provenance import environment, source_hashes, utc_now
from solarseg.submission import write_submission

DATA_ROOT = Path(os.environ.get("SOLARSEG_DATA", "/kaggle/input" if Path("/kaggle/input").exists() else "data/raw"))
WORK = Path("/kaggle/working/solarseg-run") if Path("/kaggle").exists() else ROOT / "artifacts/notebook-run"
WORK.mkdir(parents=True, exist_ok=True)
RELEASE_TAG = {a.release_tag!r}
MODEL_SHA256 = {selected["checkpoint_sha256"]!r}
PARAMS = {selected["postprocess"]!r}
TILE = {selected.get("tile", 0)}
TTA = {selected.get("tta", False)!r}
CHECKPOINT = WORK / "model.pt"
local_model = ROOT / {selected["checkpoint"]!r}
if not CHECKPOINT.exists():
    if local_model.exists():
        shutil.copy2(local_model, CHECKPOINT)
    else:
        url = f"https://github.com/{{REPO}}/releases/download/{{RELEASE_TAG}}/model.pt"
        urllib.request.urlretrieve(url, CHECKPOINT)
assert sha256(CHECKPOINT) == MODEL_SHA256, "Checkpoint checksum mismatch"
print(json.dumps(environment(), indent=2))
print("Artifact directory:", WORK)
""")
md("""
## 1. Audit inputs and freeze physical-image groups

Download with `python scripts/download_data.py` after accepting the Kaggle rules, or attach
the competition data in Kaggle. An annotation ID is not a physical image ID. All annotators,
crops, and augmentations of one observation follow the same fold. The audit uses 27-day
blocks, exact decoded-pixel hashes, and a three-day embargo on optimization data.
""")
code(f"""
data = CompetitionData(DATA_ROOT)
audit = make_manifest(data, WORK / "manifests")
assert audit["train_annotation_sha256"] == {selected["annotation_sha256"]!r}
assert audit["split_manifest_sha256"] == {selected["manifest_sha256"]!r}
manifest = pd.read_csv(WORK / "manifests/train_manifest.csv")
display(pd.Series(audit["roles"], name="Physical images").to_frame())
print("Train images / annotator records / instances / test images:",
      audit["physical_train"], audit["annotator_images"], audit["annotations"], audit["test"])
""")
md("""
## 2. Inspect input and annotator-aware target

The foreground target averages separate annotator unions. The score compares predictions
with each annotator separately. It never evaluates against this consensus target.
The example below belongs to calibration, and is selected deterministically.
""")
code("""
stem = manifest.loc[manifest.role == "calibration", "stem"].iloc[0]
image = read_image(data.image_path(stem), 1024)
target = data.soft_target(stem, 1024)
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(image, cmap="gray", vmin=0, vmax=1)
axes[0].set_title(f"H-alpha: {stem}")
axes[1].imshow(target, cmap="magma", vmin=0, vmax=1)
axes[1].set_title("Mean of annotator foreground unions")
for ax in axes: ax.axis("off")
plt.show()
""")
md("""
## 3. Verify metric and serialization contracts

PQ = sum of matched IoUs / (TP + 0.5 FP + 0.5 FN). An IoU of exactly 0.5 is not a match.
COCO uses compressed counts with column-major ordering, not ordinary Kaggle run-length pairs.
The full repository tests also compare our evaluator with the downloaded official notebook.
""")
code("""
mask = np.zeros((23, 31), dtype=np.uint8)
mask[3:11, 14:25] = 1
np.testing.assert_array_equal(mask, decode(encode(mask)))
assert counts(np.array([[0.5]]))["tp"] == 0
assert aggregate([counts(np.array([[1.0]])), counts(np.zeros((3, 0)))])["pq"] == 0.4
print("Asymmetric COCO roundtrip and pooled PQ contracts passed")
""")
md("""
## 4. Train from approved labels (opt-in)

One-channel U-Net; BCE + soft Dice; AdamW; foreground-biased crops mixed with random crops;
rotations/flips and mild intensity/blur augmentation. Training stops after the recorded update
budget. No holdout-based early stopping or external weights are used. Use a new run directory
for each experiment. Training on a different device may produce different learned weights.
""")
code(f"""
training_run = WORK / "retrained"
if RUN_TRAINING:
    train(DATA_ROOT, WORK / "manifests/train_manifest.csv", training_run,
          steps={training["steps"]}, size={training["size"]}, crop={training["crop"]},
          batch_size={training["batch_size"]}, width={training["width"]}, seed=2026)
    CHECKPOINT = training_run / "model.pt"
else:
    print("Using released checkpoint; set RUN_TRAINING=True to rerun optimization")
""")
md("""
## 5. Calibrate instance reconstruction (after optional retraining)

The recorded 25-setting grid includes the initial 16 settings and nine larger-area settings
introduced after inspecting v1 calibration false positives. This is adaptive calibration,
not an unbiased validation estimate. Thresholding follows native-resolution probability
restoration. A small closing operation and area filter precede disjoint connected components.
""")
code("""
predictor = Predictor(CHECKPOINT, device="cpu", tta=TTA, tile=TILE)
if RUN_TRAINING:
    stems = manifest.loc[manifest.role == "calibration", "stem"].tolist()
    cache = training_run / "probabilities-calibration"
    predict_cache(data, stems, predictor, cache)
    selected_result = calibrate(data, stems, cache, training_run)
    PARAMS = selected_result["params"]
else:
    print("Frozen release postprocessing:", PARAMS)
""")
md("""
## 6. Evaluate a frozen candidate (opt-in)

The original baseline was frozen before its first holdout evaluation. Those results are now
public and the holdout is no longer secret to the experimenter. Repeating this evaluation
is reproduction, not a fresh independent test. Later model-selection campaigns need nested
grouped CV or a prospectively reserved chronological block; do not tune on this cell's output.
""")
code("""
if RUN_HOLDOUT:
    output = WORK / "holdout-reproduction.json"
    if output.exists():
        raise FileExistsError("Holdout has already been evaluated in this notebook directory")
    stems = manifest.loc[manifest.role == "holdout", "stem"].tolist()
    cache = WORK / "probabilities-holdout"
    predict_cache(data, stems, predictor, cache)
    print(evaluate_and_save(data, stems, cache, PARAMS, output))
else:
    print("Holdout evaluation skipped; measured baseline results are in the repository")
""")
md("""
## 7. Infer every test observation and write a valid submission

Test labels are neither available nor required. Every test image is processed, including
images for which the model predicts zero instances. Zero detections go in the coverage
sidecar; the CSV contains only real nonempty masks. IDs preserve the original image stem.
No automatic leaderboard probing, dummy masks, or manual test corrections are performed.
""")
code("""
started = time.monotonic()
stems = sorted(data.test_paths)
cache = WORK / "probabilities-test"
predict_cache(data, stems, predictor, cache)
validation = write_submission(stems, cache, PARAMS, WORK / "submission.csv")
assert validation["images_processed"] == audit["test"]
elapsed = time.monotonic() - started
print(json.dumps(validation, indent=2))
print(f"Inference + serialization seconds (including any cache reuse): {elapsed:.1f}")
display(pd.read_csv(WORK / "submission.csv").head())
""")
md("""
## 8. Record evidence; submit only after the gates pass

The CLI gate additionally requires the protected holdout result, reviewed morphology,
official-evaluator parity, source/checkpoint/input fingerprints, and a fresh-kernel notebook
replay. `python scripts/submit.py --run <selected-run> --check-only` checks readiness and quota.
Removing `--check-only` uploads one CSV, records the attempt, and refuses blind duplicate retries.
Check the server's scoring status separately. Maximum: five/day, two final selections.

Final evaluation also requires the public source/checkpoint/notebook, the report in the host's
template, and the separate Google form. A successfully scored CSV does not complete those steps.
The exact form fields require authenticated Google access and were not verified through HTTP.
""")
code("""
proof = {"created_at": utc_now(), "completed": True, "source_commit": SOURCE_COMMIT,
         "source_hashes": source_hashes(), "checkpoint_sha256": sha256(CHECKPOINT),
         "environment": environment(), "validation": validation, "elapsed_seconds": elapsed,
         "training_rerun": RUN_TRAINING, "holdout_rerun": RUN_HOLDOUT}
(WORK / "notebook-proof.json").write_text(json.dumps(proof, indent=2))
if Path("/kaggle").exists():
    shutil.rmtree(cache)  # Generated probability maps are not needed in public outputs.
print("Validated CSV:", WORK / "submission.csv")
""")
notebook = nbf.v4.new_notebook(cells=cells)
notebook.metadata = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.12.10"},
}
for i, cell in enumerate(notebook.cells):
    cell.id = f"solarseg-{i:02d}"
path = root / "notebooks/canonical.ipynb"
nbf.write(notebook, path)
nbf.validate(nbf.read(path, as_version=4))
print(path, sha256(path))
