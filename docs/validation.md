# Validation and submission gates

## Fixed split

`artifacts/manifests/train_manifest.csv` contains one row per physical JPEG. All its annotators, crops, and augmentations follow that row. Five deterministic folds are assigned to 27-day time blocks after unioning blocks containing identical decoded pixels. Fold 0 is locked holdout; fold 1 is calibration. The remaining folds supply training observations except those within three days of calibration or holdout dates. The initial audited partition is 399 train, 149 calibration, 145 holdout, and 14 embargoed images.

This controls direct duplicate and short-range temporal leakage. It does not prove independence of solar structures that persist for longer periods or recur across rotations. Before final competition claims, compare stricter chronological and leave-one-site-out stress tests, review near-duplicate similarities, and report sensitivity to the grouping window. Do not silently change the split to obtain a better score.

The first training run found a pandas timestamp-unit mismatch in the embargo computation; it was corrected to explicit nanoseconds **before any successful model training/evaluation on that split**. A regression test now requires a usable training set and verifies the minimum time separation.

## Official metric and diagnostics

Predictions are made once per physical image and evaluated against each corresponding annotator separately. Compute every GT/prediction IoU, match when `IoU > 0.5`, pool TP IoUs and FP/FN counts over all annotator-image entries, then compute PQ. Do not average per-image PQ, merge independent annotations, or include background as a class. Our RLE implementation is numerically checked against the organizer's tensor code. Ground-truth overlaps can create multiple qualifying pairs; parity with the released evaluator is retained and prediction overlaps remain prohibited.

Report PQ, segmentation quality SQ, recognition quality RQ, positive-pair IoU/Dice distributions, one-to-many/many-to-one counts, empty detections, and per-site/year/size behavior. Distribution plots that omit zero pairs are labeled accordingly; FP/FN counts show misses. Bootstrap **physical images**, retaining all annotators together. Also report a block-bootstrap sensitivity analysis using the original time/duplicate groups. The baseline has 27 holdout groups and a block interval [0.32166, 0.36834], compared with the image interval [0.32877, 0.36327]. These fixed-model intervals do not account for every source of temporal dependence or model-selection uncertainty.

## Gate definitions

| Gate | Evidence needed | Failure action |
|---|---|---|
| G0 — Access and provenance | Official bundle; accepted rules; label and file hashes; no train/test duplicate IDs/pixels; known source licenses | Quarantine and fix input problem |
| G1 — Correctness | COCO asymmetric roundtrip, polygon conversion, exact/missing/extra/split/merge cases, strict 0.5 boundary, official evaluator parity, split integrity tests | No training claims or submission |
| G2 — Calibration | Completed checkpoint; frozen configuration; calibration grid/results; positive improvement over classical baseline; no holdout used in tuning | Diagnose model; do not use LB for tuning |
| G3 — Holdout and morphology | One frozen candidate evaluated on holdout; initial operational floor PQ 0.15; no pathological visual output; representative and worst-case examples; uncertainty | Investigate with calibration data; record holdout exposure |
| G4 — Reproduction and file validity | Notebook replay; checkpoint/config/manifest hashes; all 180 images processed; valid nonempty disjoint RLEs; deterministic CPU replay or documented platform tolerance | Fix artifact and rerun preflight |
| G5 — Submit and verify | Remaining daily quota; approved gate record bound to exact CSV hash; CLI submit success and server scoring result | Diagnose actual error; never blindly retry |
| G6 — Final entry | Select up to two final Kaggle entries; report in host template; public repository/notebook/weights; final Google form receipt; public availability through winners announcement | Entry remains administratively incomplete |

G3's 0.15 floor is only a first-submission operational threshold, **not a competitive target**. The initial competitive milestone is honest PQ around/above 0.35, followed by reproducible improvements. The hosts explicitly say 0.35 is a reasonable baseline, not the purpose or ceiling of the challenge.

## Experiment discipline

Use calibration to choose among the prespecified first baselines and tune postprocessing. The initial 16-point grid was expanded by nine area/threshold settings after excess calibration false positives were observed; record both searches. This is adaptive tuning and its maximum is optimistically biased. The holdout estimates the frozen choice independently. Do not call the calibration maximum an unbiased validation estimate.

After the first holdout result is seen, it is no longer secret to the experimenter. Do not repeatedly promote variants on it. For later campaigns, use nested grouped cross-validation or a newly reserved chronological evaluation block established before new tuning; clearly report reuse. Compare paired per-image predictions, not disconnected headline means. A default promotion requires +0.005 calibration PQ with credible morphology and runtime; for costly changes, prefer a positive paired-bootstrap interval. This is an experiment policy, not a statistical guarantee.

## Submission budget

Maximum five/day; normally use at most one initial candidate and one meaningfully improved candidate in a development cycle, preserving room for actual format failures. Query the account's limits immediately before submission. Record submission ID, time, exact artifact hash, model/config hash, description, status, score, and any server messages. Avoid leaderboard probing, filename-based answers, empty masks, public test annotations, or manual test corrections.
