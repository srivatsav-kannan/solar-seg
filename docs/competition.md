# Competition requirements and source audit

Verified **6 September 2026** from all 17 retrieved competition content pages, the separate data description, complete Rules and Foundational Rules, the organizer's self-evaluation notebook, and discussion replies obtained through the Kaggle API. Official material is cached locally in the ignored `references/official/` directory. Refresh with `python scripts/refresh_sources.py`. This document records operational requirements; the linked official rules remain authoritative.

## Task and judging

From each full-disk H-alpha observation, predict the complete extent of each individual solar filament, including barbs. Separate distinct filaments and avoid splitting one filament into many pieces. The leaderboard metric changed from Dice to **Panoptic Quality** on 7 August; the leaderboard was rescored on 12 August. Old Dice statements remain in some paragraphs, so use the updated ranking section and evaluator.

The advertised rubric is 70% quantitative (PQ, IoU/Dice distributions, fragmentation and merging) and 30% qualitative (complete method description, morphology, modular documented code). Efficiency also matters. The general foundational rules state private-leaderboard winner determination and override inconsistent competition-specific rules; the hosts explicitly describe broader scientific judging. **Record this tension and satisfy both; do not assume either leaderboard rank or a particular rubric guarantees an award.** [Overview](https://www.kaggle.com/competitions/filament-segmentation-2026/overview), [Rules](https://www.kaggle.com/competitions/filament-segmentation-2026/rules).

## Requirements register

| Requirement | Implementation / action | State |
|---|---|---|
| One registered Kaggle account; already accepted rules | CLI account `srivatsavkannan`; no alternate accounts | Verified |
| At most five team members | Keep current solo entry unless an official merger is arranged | Current solo workflow |
| At most five submissions per day | Check CLI limits immediately before every submission | Enforced by submission script |
| At most two final submissions | Select the strongest validated, reproducible candidates before close | Final selection pending |
| Final reports and solutions due 15 Nov 2026 | Displayed close: **06:00 UTC / 11:30 IST**; recheck before close | Recorded |
| Winner announcement dates differ across pages | Maintain repository access through the actual announcement; see date discrepancy below | Recorded |
| Conference 14–17 Dec 2026, Phoenix | Attendance optional; not required for prize receipt | Verified organizer reply |
| Single prediction CSV | `filament_id,segmentation_rle`; row count is variable | Implemented |
| Every predicted filament has unique ID | `<original_image_stem>_<positive_integer>` | Validated |
| Native 2048 x 2048 compressed COCO counts | No `size` column; no manually added quote characters | Validated |
| No overlapping predicted instances | One pixel belongs to at most one instance | Validated |
| No external test annotations / hand labeling test or validation | Competition training JSON only | Implemented |
| Inference from supplied H-alpha images | Grayscale JPEG input only | Implemented |
| Public Git repository, accessible without approval | [Public repository](https://github.com/srivatsav-kannan/solar-seg); MIT source license | Verified public |
| Source public immediately at close until winners announced | Keep public through winner announcement | Public now; ongoing obligation |
| Public competition code also shared on Kaggle | [Associated canonical notebook](https://www.kaggle.com/code/srivatsavkannan/solar-filaments-canonical-baseline-2026) | Public version 3 completed; exact submitted CSV reproduced |
| Exact utilized package versions | `requirements.txt` and optional development pins | Implemented |
| Notebook demonstrates entire pipeline | Canonical notebook calls reusable modules; training is an explicit opt-in | Implemented; local and Kaggle CPU inference replays passed |
| Reproduce predictions without requesting missing private files | [Checkpoint/config/checksums release](https://github.com/srivatsav-kannan/solar-seg/releases/tag/baseline-v0.1); official input acquired from Kaggle | Anonymous model download verified |
| Report in supplied Overleaf template, one PDF | Main content at most four pages, excluding Acknowledgment and References; fixed black text preserved | Draft PDF: three main pages + one acknowledgment/reference page; contact details pending |
| Final Google form, accurate team/contact/repository/report details | Form submission is separate from Kaggle CSV upload | **Pending authenticated access and final package** |
| Explain and defend AI-assisted code | `docs/understanding.md`, code walkthrough, experiment evidence | Participant preparation |
| Complete training and inference code/environment | Modules, notebook, scripts, configs, run records | Published |
| Winner license / documents | OSI-approved source license; licenses/releases and tax forms if selected | Conditional on winning |
| Banking / vendor onboarding | Valid bank capable of receiving US institutional payments; host has not specified all international details | Conditional on winning |

Sources: [Final Submission](https://www.kaggle.com/competitions/filament-segmentation-2026/overview/final-submission), [Open-access policy](https://www.kaggle.com/competitions/filament-segmentation-2026/overview/open-access-policy), [prize logistics discussion](https://www.kaggle.com/competitions/filament-segmentation-2026/discussion/727148).

## Prize and date details

The stated pool is **up to US$3,000** for the top three eligible teams. The Prizes page gives a score-squared allocation: `3000 * score_i**2 / sum(score_j**2)` over the winners, with equal division among verified members of each winning team. Payment requires AURA/NSO vendor onboarding and an account able to receive U.S. institutional transfers. Prize acceptance also grants organizers a perpetual, worldwide, non-exclusive, royalty-free license for research, education, benchmarking, archival use, and attributed scientific publications. These are terms stated by the hosts, not an assumed award or payout. [Prizes](https://www.kaggle.com/competitions/filament-segmentation-2026/overview/prizes).

There is a date discrepancy: Important Dates lists **30 November** for winner announcement, while Prizes describes announcement at the **14–17 December** conference. Keep the repository public through the actual announcement; do not automatically make it private on 30 November. The report/submission close remains the displayed 15 November deadline. [Important Dates](https://www.kaggle.com/competitions/filament-segmentation-2026/overview/important-dates).

## Actual input data

The downloaded archive contains 707 physical training JPEGs, 180 test JPEGs, and a COCO-style training JSON with 1,154 annotator-image records and 8,199 filament annotations. All images are 2048 x 2048, 8-bit grayscale; the code explicitly converts to one luminance channel. Six site suffixes are present: Bh, Ch, Lh, Mh, Th, Uh. Names encode `YYYYMMDDHHMMSSII`. Dates/sites are used for splitting and diagnostics, not as an input feature to the model.

```
MAGFiLO_1.0_Kaggle_2026/
  train/train_images/*.jpeg
  train/MAGFiLO_1.0_Annotations_kaggle2026_train.json
  test/test_images/*.jpeg
```

The JSON image key is an **annotator-prefixed string**, such as `040301-20140609195854Bh`. Several such keys may refer to the same file. Annotation IDs are UUID strings. The actual segmentation field contains polygons; `pycocotools.frPyObjects` and `merge` convert these to masks/RLE. The page describes RLE and polygons imprecisely in places; the loader follows the inspected JSON. `bbox` uses `[x,y,width,height]`; `area` is pixel area; `spine` is a polyline. Four chirality categories are Left, Right, Unidentifiable, Ambiguous; the task's output remains one filament class. Never split annotator records independently, despite the page's suggestion that they can be treated as different images. [Data description](https://www.kaggle.com/competitions/filament-segmentation-2026/data).

The full geometry audit found no malformed/nonfinite/out-of-canvas polygon arrays and no empty rasterized masks. Raster areas range from 9 to 37,739 pixels; the median is 1,228. Supplied areas match rasterized areas for all 8,199 instances. Five annotator records contain internal mask overlap, including one pair with IoU above 0.5: this makes exact organizer-metric parity consequential in the real inputs. Supplied boxes usually differ by at most one pixel, but 66 have discrepancies above two pixels and the maximum is 64.04 pixels. The current model uses polygons; derive future detector boxes from rasterized masks and inspect metadata discrepancies. Labels were not changed. See `reports/label-geometry.json`.

## Measured annotator disagreement

A calibration-only audit found 62 physical images with multiple annotation sets, yielding 142 annotator pairs. After averaging pairs within each image and weighting physical images equally, mean pairwise instance PQ was 0.33109 and foreground Dice was 0.59781. This confirms substantial disagreement in object extent and/or inclusion and motivates annotator-aware training and diagnostics. These are **agreement statistics, not model scores or a performance ceiling**; their averaging differs from competition pooling, internal annotation overlap can occur, and singly annotated observations are excluded. No test labels or newly exposed holdout labels were used. See `reports/annotator-agreement.json` and `scripts/annotator_agreement.py`.

## Actual output contract

Write one row per **predicted instance**, not one row per image or one row per known annotation. The suffix makes the row unique; it is not matched to a ground-truth index. The evaluator matches masks by overlap. Counts are compressed COCO RLE, not space-separated Kaggle run-length pairs. Encode a Fortran-contiguous `uint8` mask, then decode the bytes to ASCII. Pandas may apply normal CSV escaping; do not manually surround the field's content with quotes. Zero detections produce zero rows for that image and an explicit zero in the local coverage manifest. Verified by accepted, scored submission 56050854: 180 images processed, 178 with predictions, two with zero predictions, and 1,645 CSV rows.

The host confirmed overlapping masks are rejected. Connected-component extraction guarantees non-overlap; future Mask R-CNN/Mask2Former outputs need a deterministic mask-ownership step. Empty masks, duplicate masks, unknown stems, malformed counts, incomplete inference, and NaN probabilities fail preflight. [Submission instructions](https://www.kaggle.com/competitions/filament-segmentation-2026/overview/submission-file), [host overlap confirmation](https://www.kaggle.com/competitions/filament-segmentation-2026/discussion/735377).

## Clarifications that materially change modeling

- **27 July / 20 August:** MAGFiLO ground-truth annotations must be limited to those supplied by this competition. The full public dataset contains test annotations. Do not download it or use pretrained filament weights with unverified training provenance. [Host clarification](https://www.kaggle.com/competitions/filament-segmentation-2026/discussion/729396).
- **21 August:** `spine`, `bbox`, `area`, and `category_id` already supplied in the training dataset are permitted. This clarifies the broader-looking metadata prohibition in Overview. Mask-derived supervision is a safe initial choice. [Host reply](https://www.kaggle.com/competitions/filament-segmentation-2026/discussion/736543).
- **24 August:** external **unlabeled** GONG H-alpha images are allowed for self-supervised pretraining, with no test-label leakage and supplied-image inference. [Host reply](https://www.kaggle.com/competitions/filament-segmentation-2026/discussion/737215).
- **2 September:** original GONG FITS data is allowed for traditional image-processing approaches. This does not automatically relax the ML final inference restriction. Current pipeline avoids this ambiguity by using JPEG only. [Host reply](https://www.kaggle.com/competitions/filament-segmentation-2026/discussion/738951).
- **14 August:** some real filaments are unannotated. The hosts want scientifically useful detections while acknowledging PQ can penalize them. Report visual morphology and relation distributions alongside PQ; do not redefine the metric to favor our method. [Host reply](https://www.kaggle.com/competitions/filament-segmentation-2026/discussion/735256).
- **16 July:** AI assistance is allowed, but the participant must understand, justify, and reproduce the solution. A polished generated notebook alone is insufficient. [Organizers' remarks](https://www.kaggle.com/competitions/filament-segmentation-2026/discussion/724503).
- High displayed scores are not proof of leakage or method quality. The hosts warn about exploits and say PQ above approximately 0.35 is useful; this is a baseline reference, **not** a winning threshold. [Discussion](https://www.kaggle.com/competitions/filament-segmentation-2026/discussion/737177).

## Remaining administrative rules

Competition data is for noncommercial competition/research/education use. External tools/data must be reasonably accessible, minimally costly, appropriately licensed, and reproducible. Private code/data sharing outside an official team is prohibited. Public code sharing carries OSI-license obligations and must be associated with the Kaggle competition. Team mergers must respect size, cumulative submission count, and merger deadline.

The foundational eligibility rule requires the older of age 18 or the local age of majority, unless the sponsor has agreed and obtained appropriate guardian consent. Its listed residence exclusions are Crimea, DNR, LNR, Cuba, Iran, and North Korea, together with applicable U.S. export-control/sanctions restrictions. Employer/institution authorization is required when entering on their behalf or within employment. These are the competition’s stated criteria; the participant’s personal eligibility has not been independently verified. Direct supervision/mentorship by an organizer or other conflicts can exclude prize eligibility. Do not infer the participant's eligibility facts from a username. Prize acceptance entails source delivery, an appropriate license, technical documentation, requested eligibility/tax/release paperwork, institutional payment onboarding, and timely replies. The rules describe one-week notification response and two-week document-return windows. Taxes are the recipient's responsibility. General terms cover privacy/publicity, warranties and rights, disqualification, technical failures, rule updates, cancellation, governing law, and dispute resolution. The source MIT license does not relicense competition data.

## Read-through coverage and unresolved items

Read: overview/abstract, announcements, important dates, final submission, description, evaluation, self-evaluation, leaderboard ranking, submission file, prizes, open-access policy, organizers, sponsor, acknowledgment, data description, specific rules, and all 18 foundational-rule sections. Read all available discussion roots and replies; refreshed pagination is captured by the SDK script. The organizer notebook was pulled and its matching/aggregation code inspected and tested.

The report template is accessible through its anonymous read-only link. The **Google form requires Google authentication** through the HTTP route; its exact field list has not yet been verified. Do not mark final entry complete until it is read and a successful final submission receipt exists. The first CSV was accepted and scored by Kaggle (`COMPLETE`, public PQ 0.30); this verifies the actual submitted artifact, not every hypothetical format edge case. Separate entry/merger deadlines are not distinctly listed in the retrieved Important Dates text; recheck platform settings before any team change.
