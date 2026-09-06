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
p.add_argument("--selection", default="configs/selected.json")
p.add_argument("--output", default="notebooks/canonical.ipynb")
p.add_argument("--full-refit-plan")
a = p.parse_args()
root = Path(__file__).resolve().parents[1]
selected = json.loads((root / a.selection).read_text())
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
print("Source revision:", SOURCE_COMMIT)
"""
bootstrap = bootstrap.replace("SOURCEVALUE", repr(a.source_commit)).replace(
    "HASHVALUE", repr(source_hashes())
)
isolation = r"""
# Kaggle may preload NumPy before this cell. Never upgrade its live kernel.
# Each %%solarseg cell below runs in one persistent, clean child kernel there.
from IPython.core.magic import register_cell_magic

ISOLATED_KERNEL = Path("/kaggle").exists() or os.environ.get("SOLARSEG_ISOLATE") == "1"
if ISOLATED_KERNEL:
    import atexit
    from jupyter_client import KernelManager

    isolated_python = os.environ.get("SOLARSEG_ISOLATED_PYTHON")
    if not isolated_python:
        env_dir = Path("/tmp/solarseg-pinned-env")
        # Kaggle's Debian Python omits ensurepip; host pip can manage a pip-less venv.
        subprocess.run([sys.executable, "-m", "venv", "--without-pip", str(env_dir)], check=True)
        isolated_python = str(env_dir / "bin/python")
        pip_command = [sys.executable, "-m", "pip", "--python", isolated_python]
        # CPU build avoids downloading unused CUDA libraries for this CPU replay.
        subprocess.run([*pip_command, "install", "--quiet", "--no-cache-dir",
                        "torch==2.14.0", "--index-url", "https://download.pytorch.org/whl/cpu"], check=True)
        subprocess.run([*pip_command, "install", "--quiet", "--no-cache-dir",
                        "-r", str(ROOT / "requirements.txt")], check=True)
    kernel_manager = KernelManager(kernel_name="python3")
    kernel_manager.kernel_spec.argv = [isolated_python, "-m", "ipykernel_launcher", "-f", "{connection_file}"]
    child_env = dict(os.environ)
    child_env.pop("PYTHONPATH", None)
    child_env.pop("PYTHONHOME", None)
    kernel_manager.start_kernel(cwd=str(ROOT), env=child_env)
    kernel_client = kernel_manager.blocking_client()
    kernel_client.start_channels()
    kernel_client.wait_for_ready(timeout=120)
    atexit.register(lambda: kernel_manager.shutdown_kernel(now=True))

    def execute_isolated(source):
        reply = kernel_client.execute_interactive(source, timeout=7200, allow_stdin=False)
        if reply["content"]["status"] != "ok":
            raise RuntimeError("Isolated cell failed: " + str(reply["content"]))

    execute_isolated(COMMON_INIT + f"\nRUN_TRAINING={RUN_TRAINING!r}\nRUN_HOLDOUT={RUN_HOLDOUT!r}")

@register_cell_magic
def solarseg(line, cell):
    if ISOLATED_KERNEL:
        execute_isolated(cell)
    else:
        result = get_ipython().run_cell(cell)
        result.raise_error()
"""
code(bootstrap + "\nCOMMON_INIT = " + repr(bootstrap) + "\n" + isolation)
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
if a.full_refit_plan:
    refit = json.loads((root / a.full_refit_plan).read_text())
    assert len(training["train_stems"]) == 707
    assert training["manifest_sha256"] == refit["manifest_sha256"]
    assert sha256(root / selected["checkpoint"]) == selected["checkpoint_sha256"]
    assert all(training[k] == v for k, v in refit["recipe"].items())
    assert a.release_tag != "baseline-v0.1", "Use the new refit checkpoint release"
    for cell in cells:
        s = cell.source
        if cell.cell_type == "markdown":
            if s.startswith("# Solar filaments:"):
                s = """# Solar filaments: full-data refit of the validated recipe

This executable notebook refits the previously selected U-Net recipe on all **707 approved
training observations**, or reproduces its released predictions. The default run loads the
checkpoint, audits inputs, checks the mask contract, and processes all 180 test images on CPU.

The original 399-image checkpoint achieved protected-holdout PQ 0.34625. That number supports
the selected recipe; **it is not a held-out score for this refit**. Its former calibration and
holdout images are now training observations. The original selection workflow and evidence
remain in the [parent notebook](https://www.kaggle.com/code/srivatsavkannan/solar-filaments-canonical-baseline-2026)
and [repository](https://github.com/srivatsav-kannan/solar-seg).

Set `RUN_TRAINING=True` for scratch training with the fixed recipe. Inference replay alone
does not mean training was rerun. Calibration and holdout evaluation are disabled for this
all-data checkpoint. No automatic competition submission occurs. Use only the competition
training annotations; the full public MAGFiLO label archive overlaps the test set.
"""
            elif s.startswith("## 1."):
                s += "\n\nThe original split is reproduced for provenance. A separate manifest then assigns all 707 approved observations to final training."
            elif s.startswith("## 2."):
                s = s.replace(
                    "The example below belongs to calibration, and is selected deterministically.",
                    "The deterministic example belonged to the original calibration split and is now part of final training; this illustration is not independent evaluation.",
                )
            elif s.startswith("## 5."):
                s = """## 5. Apply the previously frozen reconstruction recipe

Threshold, area filtering, closing, and four-flip averaging are fixed from the parent
selection. Do not tune them on observations used to train this refit. New modeling choices
require the separate grouped comparison protocol. Restore probabilities to native resolution
before thresholding, then extract disjoint connected components.
"""
            elif s.startswith("## 6."):
                s = """## 6. Keep validation provenance separate from final fitting

The original protected-holdout result applies to the original 399-image checkpoint. This
checkpoint trains on all 707 images and has no independent internal holdout. The parent
notebook documents selection and optional reproduction of its already-exposed holdout.
Never report a score measured on this refit's own training images as held-out validation.
"""
            elif s.startswith("## 8."):
                s = """## 8. Record evidence and submit only the checked artifact

This refit needs its own gate record: exact parent recipe/source equivalence, all approved
training observations, checkpoint/input hashes, full CPU inference, a fresh notebook replay,
and a valid CSV with complete coverage. The original checkpoint-specific gate does not apply
to these new weights. Check the five-per-day quota and save the server receipt separately.

The final competition entry also requires public code/checkpoints/notebook, the specified
report, and the authenticated organizer form. This notebook does not submit the CSV or form.
"""
        else:
            s = s.replace('"/kaggle/working/solarseg-run"', '"/kaggle/working/solarseg-refit-run"')
            s = s.replace('"artifacts/notebook-run"', '"artifacts/notebook-refit-run"')
            if s.startswith("import time"):
                s += '\nassert not RUN_HOLDOUT, "The all-data refit has no independent holdout"\n'
            elif s.startswith("data = CompetitionData"):
                s += f"""
full_manifest = manifest.copy()
full_manifest["role"] = "train"
full_manifest_path = WORK / "full-training-manifest.csv"
full_manifest.to_csv(full_manifest_path, index=False)
assert len(full_manifest) == 707 and set(full_manifest.stem) == set(data.by_stem)
assert sha256(full_manifest_path) == {refit["manifest_sha256"]!r}
print("Final training observations:", len(full_manifest))
"""
            elif s.startswith("training_run ="):
                s = s.replace(
                    'training_run = WORK / "retrained"',
                    'training_run = Path("/tmp/solarseg-refit-training/retrained") if Path("/kaggle").exists() else WORK / "retrained"',
                ).replace('WORK / "manifests/train_manifest.csv"', "full_manifest_path")
                s += """
if RUN_TRAINING and Path("/kaggle").exists():
    published_training = WORK / "retrained"
    published_training.mkdir(exist_ok=False)
    for name in ["model.pt", "config.json", "history.json"]:
        shutil.copy2(training_run / name, published_training / name)
    CHECKPOINT = published_training / "model.pt"
"""
            elif s.startswith("predictor = Predictor"):
                s = """predictor = Predictor(CHECKPOINT, device="cpu", tta=TTA, tile=TILE)
print("Frozen parent-recipe postprocessing:", PARAMS)
"""
            elif s.startswith("if RUN_HOLDOUT:"):
                s = 'print("No independent holdout evaluation is available for the all-data refit")'
            elif s.startswith("proof ="):
                insertion = f"""proof.update(full_data_refit=True,
    validation_scope={refit["validation_scope"]!r},
    parent_checkpoint_sha256={refit["parent_checkpoint_sha256"]!r},
    training_manifest_sha256=sha256(full_manifest_path))
"""
                s = s.replace(
                    '(WORK / "notebook-proof.json").write_text',
                    insertion + '(WORK / "notebook-proof.json").write_text',
                )
        cell.source = s.strip()

notebook = nbf.v4.new_notebook(cells=cells)
first_code = True
for cell in notebook.cells:
    if cell.cell_type == "code":
        if first_code:
            first_code = False
        else:
            cell.source = "%%solarseg\n" + cell.source
notebook.metadata = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.12.10"},
}
for i, cell in enumerate(notebook.cells):
    cell.id = f"solarseg-{i:02d}"
path = root / a.output
path.parent.mkdir(parents=True, exist_ok=True)
nbf.write(notebook, path)
nbf.validate(nbf.read(path, as_version=4))
print(path, sha256(path))
