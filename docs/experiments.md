# Experiment ledger

Research began on 6 September 2026. All numbers below are measured on our split; calibration scores are used for tuning and are optimistic. No external weights or labels are used.

| Run | Training | Calibration PQ | Status |
|---|---|---:|---|
| classical-v1 | Local-background intensity deficit, no learning | 0.08265 | Reference; 16 postprocessing settings |
| unet-v1 | Width 16, 1024-pixel disk, 256 crops, batch 6, 3,000 updates | 0.25096 | Initial 16-point search |
| unet-v1 / expanded | Same checkpoint; nine larger-area settings added | 0.27503 | Threshold 0.60, area 800 native pixels, closing radius 3 |
| unet-v2 | Width 24, 384 crops, 6,000 updates; otherwise same protocol | Pending | Training underway |

The expanded search responds to excess false positives and fragments in calibration. It is documented adaptive tuning, not a new independent validation set. The maximum cannot be interpreted as an unbiased estimate of generalization.

The protected holdout has not yet been used to select a model. Candidate selection must be written to `configs/selected.json` before opening holdout results. Later changes must record any holdout exposure.

## Reproduction and provenance

- Split: 399 optimization / 149 calibration / 145 holdout / 14 embargoed physical images. Seed 2026.
- Split SHA-256: `ebec49b919b111f0591f1740cc10a470c127371203e7522c683c9ec5a34283b8`.
- Label SHA-256: `5da9e92b5a1a1947fd5d57adb6688269625c48ec1ef884daf2a01618c9ed54a1`.
- Local hardware: Apple Silicon, 24 GB unified memory, PyTorch MPS. Runtime is measured after cache preparation and excludes downloads.
- Run directories contain config, loss history, checkpoint/hash, prediction caches, calibration results, and later evaluation/submission records. The baseline training source snapshot was captured retrospectively, before changing the implementation; this is not a claim of a launch-time Git commit.
- An earlier width-16 process was interrupted before its final checkpoint; it was discarded and the 3,000-update experiment was rerun from scratch. No partial checkpoint was scored.
- A pandas timestamp-unit defect was corrected before successful training. Regression tests now verify the three-day embargo and a nonempty optimization split.

## Next experiments

Compare full-frame and overlapping-tile inference on calibration to measure any shift caused by patch-trained GroupNorm. Then compare larger context/capacity. Freeze the best measured candidate, examine representative and failure cases, evaluate holdout once, reproduce predictions on CPU, and submit only when the recorded gates pass.

Further campaigns should prioritize native detail, boundary/affinity supervision, detector-plus-refiner instance models, and complementary ensembles. These are research plans, not implemented results. See [the literature review](literature-review.md) for evidence and limitations.
