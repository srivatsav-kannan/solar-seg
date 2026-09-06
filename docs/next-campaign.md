# Next model-development campaign

This is a prospective research plan, not a description of completed experiments. The first public PQ is 0.30; the original holdout has been exposed. The measured errors justify testing detail, context, and instance reconstruction before scaling an unvalidated transformer.

## Validation before training

Keep the official input hashes and original physical-image/time groups. Reserve all test images exclusively for automatic inference. The nested grouped cross-validation manifest index is now frozen in `configs/nested-cv.json`, before any nested-CV model training: five outer folds and three inner folds within each eligible outer training partition. A three-day embargo excludes outer-neighbor observations from both inner optimization and calibration; optimization also stays more than three days from its inner calibration observations. Fit every outer-fold checkpoint from scratch without that fold's labels. Choose thresholds within its inner calibration partition, then pool the outer-fold annotator counts for the overall PQ; never average fold PQ values as the main metric.

Earlier models and analysis used parts of this same corpus. Label this **subsequent development cross-validation**, not a newly untouched test set. Predeclare a stricter chronological stress split and a leave-one-site-out diagnostic before viewing those results. Run a near-duplicate similarity audit in addition to exact-pixel matching. Report all these results, including disagreements, rather than selecting the split that looks best. A genuinely independent future generalization claim needs previously unseen observations and permitted labels.

The deterministic generator is `scripts/make_nested_folds.py`. It creates 15 inner-fit manifests and five outer-refit manifests under `artifacts/nested-cv/`. The public index stores every file hash and role count; the input manifest is already public in `configs/manifests/`. To regenerate into fresh paths:

```bash
python scripts/make_nested_folds.py --output artifacts/nested-cv-copy --index artifacts/nested-cv-copy-index.json
```

The new index timestamp/path differ, while per-manifest hashes match. Automated tests verify outer-group exclusion, duplicate integrity, the actual date gaps, and one inner-calibration assignment per eligible physical observation. No nested-CV models have been trained yet.

## Bounded comparison

| Order | Candidate | Isolated question | Starting budget |
|---|---|---|---|
| 0 | Refit the present width-24 baseline in the new folds | Establish the comparison under identical folds | Existing 6,000-update recipe per fit |
| 1 | Native 2048-pixel sampling, 512-pixel overlapping inference tiles | Does retained detail improve masks and PQ? | Same width, optimizer, sample count; record extra runtime |
| 2 | Coarse 1024-disk context plus native local refinement | Can whole-object context reduce large-mask truncation? | Fixed backbone and training budget; compare with candidate 1 |
| 3 | Foreground + mask-derived boundary/affinity heads | Can reconstruction reduce fragmentation/merging? | One prespecified auxiliary weight, then a bounded inner-calibration search |
| 4 | Mask R-CNN or CondInst with high-resolution refinement | Does direct instance identity outperform components? | Scratch initialization unless external provenance is fully audited |

The first exploratory native-resolution screen is frozen in `configs/native-screen.json`: 2048-pixel images, the same 384-pixel crops, width 24, 6,000 updates, and full-frame versus 384-pixel tiled inference without TTA. This isolates linear sampling resolution at matched crop dimensions/update count, while halving physical context per crop; the larger 512-pixel context variant remains a later experiment.

Screen candidates on the original calibration partition only as exploratory development, with no new holdout claims. Advance at most two to the expensive nested comparison. Use identical seeds, folds, image sampling, and postprocessing search budgets for paired comparisons. Change context, architecture, and update count separately where the purpose is an ablation. Record peak memory and complete inference time; do not compare a cached prediction run with a cold run.

The present soft consensus targets average annotators. A separate, low-cost comparison can test random-annotator hard targets against that choice, with sampling and all other settings held fixed. Avoid multiplying architectural and label-policy choices into an uncontrolled search.

## Promotion and submission

Require pooled cross-validation PQ improvement of at least 0.005 over the identically refitted baseline, a positive paired block-bootstrap interval where feasible, and reviewed morphology across sites and size groups. Report small/large-mask recall and merge/fragment diagnostics even when PQ improves. For a costly model or ensemble, require an explicit accuracy/runtime tradeoff. Ensemble only candidates whose paired errors show complementarity, then calibrate the ensemble within the inner folds.

Freeze the final configuration and fold ensemble before test inference. A final refit using all approved training observations is allowed only after selection, with hyperparameters fixed and training provenance recorded; it must not be represented as having the old holdout validation. Create a new campaign selection record and corresponding gates rather than overwriting `configs/selected.json` or reusing the first model's G3 evidence.

Run the current canonical workflow with the new release, validate all 180 outputs, review a systematic morphology sample, and submit one improved candidate through the CLI. Public leaderboard movement is external feedback, not a threshold-search loop. Preserve the existing accepted baseline and leave room within five daily submissions for genuine format failures. Decide the two final entries only after reproducibility and the final report/form package are complete.
