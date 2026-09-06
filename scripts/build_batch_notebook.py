"""Generate the two-stage batch-Dice workflow after its paired CV gate passes."""

import argparse
import json
from pathlib import Path

import nbformat

from solarseg.data import sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--release-tag", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    selected = json.loads((root / args.selection).read_text())
    plan = json.loads((root / "configs/batch-dice-full-fit-02.json").read_text())
    comparison_path = root / "artifacts/cv-02/paired-batch-dice/comparison.json"
    comparison = json.loads(comparison_path.read_text())
    assert comparison["statistical_gates_passed"]
    assert sha256(root / selected["checkpoint"]) == selected["checkpoint_sha256"]
    assert selected["postprocess"] == plan["inference"]["postprocess"]
    template = root / "notebooks/full_refit.ipynb"
    assert sha256(template) == "8729fb87ae7203c4b7d2950b714d48e375aa85cbc733afc81a4d923f4ec884a0"
    notebook = nbformat.read(template, as_version=4)
    for cell in notebook.cells:
        cell.source = cell.source.replace(
            "4cc77325af2e4bebba1de4f30f489db72ea35478", args.source_commit
        )
        cell.source = cell.source.replace(
            "/kaggle/working/solarseg-refit-run", "/kaggle/working/solarseg-batch-run"
        )
        cell.source = cell.source.replace(
            "artifacts/notebook-refit-run", "artifacts/notebook-batch-run"
        )
    notebook.cells[0].source = """# Solar filaments: batch Dice and additional training

This notebook reproduces the released all-data model's predictions, or runs its two-stage
training procedure. A grayscale H-alpha image is mapped to filament foreground probabilities;
native-resolution connected components produce disjoint instance masks. The score is pooled
per-annotator Panoptic Quality with strict IoU > 0.5.

Stage one fits the original U-Net recipe for 6,000 updates. Stage two uses batch Dice plus BCE
for 3,000 additional updates at a smaller learning rate. The comparison tests both changes
together; it does not isolate the loss from the extra training. Only approved competition
training labels are used. The full public MAGFiLO archive contains test labels and is prohibited.

The recipe has a separately frozen three-fold subsequent-development comparison. All 707
approved images train the released model, so it has no independent internal holdout score.
The old baseline's 0.34625 holdout PQ does not apply to this model. Shared inner calibration and
earlier work on the corpus limit independence; see the exact comparison record below.

The default run loads weights and performs fresh CPU inference. Set `RUN_TRAINING=True` to
fine-tune the released, approved stage-one checkpoint. Also set `RETRAIN_PARENT=True` in the
training cell to refit stage one from scratch. Inference replay alone is not retraining.
No competition submission occurs automatically.
"""
    setup = notebook.cells[2].source
    setup = setup.replace("RELEASE_TAG = 'full-refit-v0.2'", f"RELEASE_TAG = {args.release_tag!r}")
    setup = setup.replace(
        "4f96bc349e6e6c7bd22fa6921071eb4ad5a3b90c88737ba4c4c02e9a1e246a08",
        selected["checkpoint_sha256"],
    )
    setup = setup.replace("artifacts/unet-full-v1/model.pt", selected["checkpoint"])
    setup += "\nfrom research.train import train as train_refiner\n"
    notebook.cells[2].source = setup
    notebook.cells[9].source = """## 4. Train from approved labels (opt-in)

The architecture is the same width-24 U-Net. Stage one uses BCE plus per-image Dice for
6,000 updates at initial learning rate 0.001. Stage two uses BCE plus Dice aggregated across
the batch for 3,000 updates at initial learning rate 0.0002. Both use 1024-pixel disks,
384-pixel crops, batch size six, AdamW, cosine decay, the same augmentations, and seed 2026.

Batch Dice changes how empty crops contribute to the loss. Training code lives in reusable
modules. Temporary Kaggle training caches stay outside published outputs. Saved configurations
record the actual numerical environment; CUDA training and local MPS training are not assumed
bitwise equivalent. The released inference checkpoint has an exact checksum.
"""
    notebook.cells[10].source = f"""%%solarseg
RETRAIN_PARENT = False
training_root = Path("/tmp/solarseg-batch-training") if Path("/kaggle").exists() else WORK / "training"
parent_run = training_root / "parent"
fine_run = training_root / "refined"
if RUN_TRAINING:
    training_root.mkdir(parents=True, exist_ok=True)
    if RETRAIN_PARENT:
        train(DATA_ROOT, full_manifest_path, parent_run,
              steps=6000, size=1024, crop=384, batch_size=6, width=24, seed=2026)
        parent_checkpoint = parent_run / "model.pt"
    else:
        parent_checkpoint = training_root / "released-parent.pt"
        if not parent_checkpoint.exists():
            urllib.request.urlretrieve(
                f"https://github.com/{{REPO}}/releases/download/full-refit-v0.2/model.pt",
                parent_checkpoint)
        assert sha256(parent_checkpoint) == {plan["initialize_sha256"]!r}
    train_refiner(DATA_ROOT, full_manifest_path, fine_run,
                  steps=3000, size=1024, crop=384, batch_size=6, width=24,
                  depth=3, seed=2026, learning_rate=0.0002,
                  initialize=parent_checkpoint, loss_mode="batch")
    CHECKPOINT = fine_run / "model.pt"
    if Path("/kaggle").exists():
        published = WORK / "retrained"
        published.mkdir(exist_ok=False)
        for name in ["model.pt", "config.json", "history.json"]:
            shutil.copy2(fine_run / name, published / name)
        CHECKPOINT = published / "model.pt"
print("Training rerun:", RUN_TRAINING, "Parent refit:", RUN_TRAINING and RETRAIN_PARENT)
"""
    notebook.cells[11].source = """## 5. Apply the frozen reconstruction settings

Four-flip averaging, threshold 0.6, minimum area 400 native pixels, and closing radius 3
were fixed from development calibration before the outer comparison. The all-data fit's
training observations must not be used to retune them. Restore probabilities to 2048 x 2048
before thresholding. Small components are filtered; retained masks are nonempty and disjoint.
"""
    notebook.cells[12].source = """%%solarseg
predictor = Predictor(CHECKPOINT, device="cpu", tta=TTA, tile=TILE)
print("Frozen reconstruction settings:", PARAMS)
"""
    notebook.cells[13].source = """## 6. Inspect the frozen paired validation evidence

Three outer folds compare separately trained baseline and fine-tuned models. Each fit's
thresholds are chosen on its own inner calibration partition, then frozen before any outer
inference. All annotators of a physical image and all crops stay together; time groups and a
three-day embargo separate partitions. Outer counts are pooled, and temporal groups are
resampled for paired uncertainty. The common inner-calibration partition limits independence.

The exact models, manifests, training commands, and evaluation code are in the repository's
`configs/batch-dice-cv-02.json`, `notebooks/batch-dice-cv-02/`, and
`research/evaluate_batch_cv.py`. The downloaded record below is historical validation evidence,
not a claim that this notebook reruns cross-validation. It does not evaluate the all-data
checkpoint on its own training images.
"""
    notebook.cells[14].source = f"""%%solarseg
CV_SHA256 = {sha256(comparison_path)!r}
CV_RECORD = WORK / "cv-comparison.json"
local_cv = ROOT / "artifacts/cv-02/paired-batch-dice/comparison.json"
if not CV_RECORD.exists():
    if local_cv.exists():
        shutil.copy2(local_cv, CV_RECORD)
    else:
        urllib.request.urlretrieve(
            f"https://github.com/{{REPO}}/releases/download/{{RELEASE_TAG}}/cv-comparison.json",
            CV_RECORD)
assert sha256(CV_RECORD) == CV_SHA256
cv = json.loads(CV_RECORD.read_text())
assert cv["statistical_gates_passed"]
print(json.dumps({{key: cv[key] for key in ["pq_gain", "paired_block_bootstrap_95ci", "fold_pq", "scope"]}}, indent=2))
"""
    notebook.cells[17].source = """## 8. Record evidence and submit only the checked file

Submission requires the paired CV gate, morphology review, the approved full-data training
recipe, exact checkpoint and input fingerprints, a fresh CPU notebook replay, and a valid CSV
covering all 180 test observations. Check quota immediately before uploading and retain the
server receipt. The notebook never uploads automatically. The competition's report and
authenticated final form remain separate final-entry obligations.
"""
    notebook.cells[18].source = f"""%%solarseg
proof = {{"created_at": utc_now(), "completed": True, "source_commit": SOURCE_COMMIT,
         "source_hashes": source_hashes(), "checkpoint_sha256": sha256(CHECKPOINT),
         "environment": environment(), "validation": validation, "elapsed_seconds": elapsed,
         "training_rerun": RUN_TRAINING, "holdout_rerun": False, "full_data_refit": True,
         "training_manifest_sha256": sha256(full_manifest_path),
         "cv_comparison_sha256": sha256(CV_RECORD), "recipe": "baseline then batch-Dice fine-tuning",
         "validation_scope": {plan["validation_scope"]!r}}}
(WORK / "notebook-proof.json").write_text(json.dumps(proof, indent=2))
if Path("/kaggle").exists():
    shutil.rmtree(cache)
print("Validated CSV:", WORK / "submission.csv")
"""
    for cell in notebook.cells:
        if cell.cell_type == "code":
            cell.execution_count = None
            cell.outputs = []
    output = root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(notebook, output)
    print(output, sha256(output))


if __name__ == "__main__":
    main()
