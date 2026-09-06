# Competition requirements and source audit

Verified **6 September 2026** from all 19 competition content pages, the complete Rules and Foundational Rules, the organizer's self-evaluation notebook, and discussion replies obtained through the Kaggle API. Official material is cached locally in the ignored `references/official/` directory. Refresh with `python scripts/refresh_sources.py`. This document records operational requirements; the linked official rules remain authoritative.

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
| Winner announcement 30 Nov 2026 | Maintain repository access through at least this date | Recorded |
| Conference 14–17 Dec 2026, Phoenix | Attendance optional; not required for prize receipt | Verified organizer reply |
| Single prediction CSV | `filament_id,segmentation_rle`; row count is variable | Implemented |
| Every predicted filament has unique ID | `<original_image_stem>_<positive_integer>` | Validated |
| Native 2048 x 2048 compressed COCO counts | No `size` column; no manually added quote characters | Validated |
| No overlapping predicted instances | One pixel belongs to at most one instance | Validated |
| No external test annotations / hand labeling test or validation | Competition training JSON only | Implemented |
| Inference from supplied H-alpha images | Grayscale JPEG input only | Implemented |
| Public Git repository, accessible without approval | Owner's GitHub; MIT source license | Release workflow |
| Source public immediately at close until winners announced | Keep public throughout once published | Release workflow |
| Public competition code also shared on Kaggle | Publish associated canonical notebook | Required release step |
| Exact utilized package versions | `requirements.txt` and optional development pins | Implemented |
| Notebook demonstrates entire pipeline | Canonical notebook calls reusable modules | Required release step |
| Reproduce predictions without requesting missing private files | Publish checkpoint/config/checksums with source; official input acquired from Kaggle | Release gate |
| Report in supplied Overleaf template, one PDF | Main content at most four pages, excluding Acknowledgment and References; fixed black text preserved | Final package gate |
| Final Google form, accurate team/contact/repository/report details | Form submission is separate from Kaggle CSV upload | **Pending authenticated access and final package** |
| Explain and defend AI-assisted code | `docs/understanding.md`, code walkthrough, experiment evidence | Participant preparation |
| Complete training and inference code/environment | Modules, notebook, scripts, configs, run records | Required release step |
| Winner license / documents | OSI-approved source license; licenses/releases and tax forms if selected | Conditional on winning |
| Banking / vendor onboarding | Valid bank capable of receiving US institutional payments; host has not specified all international details | Conditional on winning |

Sources: [Final Submission](https://www.kaggle.com/competitions/filament-segmentation-2026/overview/final-submission), [Open-access policy](https://www.kaggle.com/competitions/filament-segmentation-2026/overview/open-access-policy), [prize logistics discussion](https://www.kaggle.com/competitions/filament-segmentation-2026/discussion/727148).

## Actual input data

The downloaded archive contains 707 physical training JPEGs, 180 test JPEGs, and a COCO-style training JSON with 1,154 annotator-image records and 8,199 filament annotations. All images are 2048 x 2048, 8-bit grayscale; the code explicitly converts to one luminance channel. Six site suffixes are present: Bh, Ch, Lh, Mh, Th, Uh. Names encode `YYYYMMDDHHMMSSII`. Dates/sites are used for splitting and diagnostics, not as an input feature to the model.

```
MAGFiLO_1.0_Kaggle_2026/
  train/train_images/*.jpeg
  train/MAGFiLO_1.0_Annotations_kaggle2026_train.json
  test/test_images/*.jpeg
```

The JSON image key is an **annotator-prefixed string**, such as `040301-20140609195854Bh`. Several such keys may refer to the same file. Annotation IDs are UUID strings. The actual segmentation field contains polygons; `pycocotools.frPyObjects` and `merge` convert these to masks/RLE. The page describes RLE and polygons imprecisely in places; the loader follows the inspected JSON. `bbox` uses `[x,y,width,height]`; `area` is pixel area; `spine` is a polyline. Four chirality categories are Left, Right, Unidentifiable, Ambiguous; the task's output remains one filament class. Never split annotator records independently, despite the page's suggestion that they can be treated as different images. [Data description](https://www.kaggle.com/competitions/filament-segmentation-2026/data).

## Actual output contract

Write one row per **predicted instance**, not one row per image or one row per known annotation. The suffix makes the row unique; it is not matched to a ground-truth index. The evaluator matches masks by overlap. Counts are compressed COCO RLE, not space-separated Kaggle run-length pairs. Encode a Fortran-contiguous `uint8` mask, then decode the bytes to ASCII. Pandas may apply normal CSV escaping; do not manually surround the field's content with quotes. Zero detections produce zero rows for that image and an explicit zero in the local coverage manifest. This behavior still needs server verification on the first real submission.

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

Eligibility includes account registration, applicable age/majority requirements, jurisdiction/sanctions restrictions, and employer/institution authorization where relevant. Direct supervision/mentorship by an organizer or other conflicts can exclude prize eligibility. Do not infer the participant's eligibility facts from a username. Prize acceptance entails source delivery, an appropriate license, technical documentation, requested eligibility/tax/release paperwork, institutional payment onboarding, and timely replies. The rules describe one-week notification response and two-week document-return windows. Taxes are the recipient's responsibility. General terms cover privacy/publicity, warranties and rights, disqualification, technical failures, rule updates, cancellation, governing law, and dispute resolution. The source MIT license does not relicense competition data.

## Read-through coverage and unresolved items

Read: overview/abstract, announcements, important dates, final submission, description, evaluation, self-evaluation, leaderboard ranking, submission file, prizes, open-access policy, organizers, sponsor, acknowledgment, data description, specific rules, and all 18 foundational-rule sections. Read all available discussion roots and replies; refreshed pagination is captured by the SDK script. The organizer notebook was pulled and its matching/aggregation code inspected and tested.

The report template is accessible through its anonymous read-only link. The **Google form requires Google authentication** through the HTTP route; its exact field list has not yet been verified. Do not mark final entry complete until it is read and a successful final submission receipt exists. The platform does not expose all server-side edge checks in the self-evaluation notebook, so the first valid CSV must be checked for server acceptance. Separate entry/merger deadlines are not distinctly listed in the retrieved Important Dates text; recheck platform settings before any team change.
