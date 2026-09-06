"""Fill the organizer's template with measured results, then compile LaTeX."""

import json
import shutil
import subprocess
from pathlib import Path

from PIL import Image

root = Path(__file__).resolve().parents[1]
template_dir = root / "references/official/report-template"
template = (template_dir / "main.tex").read_text()
selected = json.loads((root / "configs/selected.json").read_text())
run = root / selected["run"]
h = json.loads((run / "holdout.json").read_text())
cfg = json.loads((root / selected["training_run"] / "config.json").read_text())
screen = json.loads((root / "reports/native-screen.json").read_text())
block_ci = json.loads((root / "reports/temporal-bootstrap.json").read_text())["block_bootstrap_95ci"]
author_path = root / "configs/author.json"
author = (
    json.loads(author_path.read_text()) if author_path.exists() else {"name": "Srivatsav Kannan"}
)
out = root / "reports/source"
out.mkdir(parents=True, exist_ok=True)
pdf_out = root / "output/pdf"
pdf_out.mkdir(parents=True, exist_ok=True)
build = root / "reports/report-build"
build.mkdir(parents=True, exist_ok=True)


def tex_escape(text):
    return str(text).replace("&", r"\&").replace("_", r"\_").replace("%", r"\%")


prefix = template[: template.index(r"\begin{document}")].replace(
    r"\guideStyletrue", r"\guideStylefalse"
)
prefix += "\n\\setlength{\\emergencystretch}{1em}\n"
intro = template.split("\\section{Introduction}\n")[1].split(r"\guide{")[0]
acknowledgment = template[template.index(r"\section{Acknowledgment}") :]
body = r"""
\begin{document}
\title{A Reproducible U-Net Baseline with Grouped Validation}
\subtitle{A Solution to the Solar Filament Segmentation Challenge 2026}
\author{AUTHOR}
AUTHORDETAILS
\begin{abstract}
This report describes our solution to the Solar Filament Segmentation Challenge~2026, a Kaggle competition on automatic segmentation of solar filaments in GONG H-$\alpha$ observations.
We train a compact, single-channel U-Net from scratch using only competition-approved labels. Independent annotator unions form soft foreground targets; annotators remain separate in the official instance-aware evaluation. Physical observations are grouped into temporal folds with an embargo. Calibration determines the model and native-resolution instance reconstruction before a protected holdout is opened. The frozen baseline achieves holdout Panoptic Quality (PQ) of HOLDOUTPQ (physical-image bootstrap 95\% interval: CILOW--CIHIGH) and Kaggle public PQ 0.30. We provide modular code, exact package versions, a canonical notebook, checkpoints, visual diagnostics, and submission checks. This is a development baseline with substantial performance headroom.
\end{abstract}
\maketitle
\section{Introduction}
FIXEDINTRO
We report data provenance, annotator dependence, instance errors, and evaluation uncertainty for a reproducible baseline. All reported experiments were run on 6 September 2026. Final contact details and the separate final-entry form remain pending.

\section{Methodology}\label{sec:methodology}
\subsection{Inputs, annotation policy, and splits}
The supplied bundle contains 707 physical training JPEGs and 180 test JPEGs, all $2048\times2048$ grayscale pixels. There are 1,154 annotator-image records and 8,199 instance polygons. Each file can have several independent annotators; splitting these records separately would leak the same observation across partitions. The four chirality categories do not change the single-class output contract.

We use only \texttt{MAGFiLO\_1.0\_Annotations\_kaggle2026\_train.json}. The full public MAGFiLO annotation release overlaps competition test observations and is excluded. No external training images, pretrained weights, manually corrected test masks, filename-based predictions, or auxiliary inference metadata are used. The annotation file and every input file are SHA-256 hashed. Decoded pixels are additionally hashed to find exact duplicates; none were found across train and test.

We assign physical observations to five deterministic grouped folds (seed 2026), using 27-day blocks anchored at 1 January 2010. Blocks connected by exact pixel duplicates would be united. Fold 0 is holdout, fold 1 calibration, and the other folds are optimization data after removing observations within three days of either evaluation partition. The resulting counts are 399 training, 149 calibration, 145 holdout, and 14 embargoed images. This protocol reduces short-range leakage; it does not prove independence of structures persisting beyond the embargo or recurring across rotations. The held-out images span the six GONG site suffixes present in the bundle.

\subsection{Targets and preprocessing}
For each annotator, polygon masks are rasterized with the COCO API and combined into a binary foreground union. These independent unions are averaged, giving a per-pixel soft target that retains disagreement instead of treating annotators as separate training samples. BOX downsampling preserves partial coverage; targets are cached as 8-bit probabilities. Evaluation never averages or unions annotator ground truths.

JPEG intensities are converted to one luminance channel, scaled to $[0,1]$, resized bilinearly to $1024\times1024$, then normalized by $(x-0.5)/0.25$. We do not apply additional limb flattening: the dataset's JPEG generation already modifies large-scale intensity variation. Half the crops are sampled around a foreground location and half uniformly; physical images are sampled uniformly. Rotations by multiples of 90 degrees, flips, modest gain/offset perturbations, and occasional weak Gaussian blur provide augmentation. No test-time visual edits are made.

\subsection{Model and optimization}
The baseline follows U-Net's encoder--decoder and skip-connection design~\cite{unet}. At each scale, two $3\times3$ convolutions use GroupNorm and SiLU. Three downsampling stages yield widths $w,2w,4w,8w$; bilinear upsampling and concatenated skips restore resolution. A $1\times1$ head produces one foreground logit. The selected model uses width WIDTH and CROPSIZE-pixel crops. A second compact configuration provides a calibration comparator; all variants use the same physical-image split.

We minimize binary cross-entropy plus soft Dice loss with AdamW, learning rate $10^{-3}$, weight decay $10^{-4}$, a cosine schedule to $10^{-5}$, batch size six, and gradient clipping at one. Training stops after STEPS updates chosen before holdout inspection. The selected run took TRAINSECONDS seconds on local Apple Silicon with 24 GB unified memory, excluding input preparation. This time is specific to this hardware and implementation, and is not a benchmark against published models. Seeds, configuration, loss history, checkpoint checksum, and environment versions accompany the release.

\begin{figure*}[t]
\centering
\includegraphics[width=\textwidth]{../figures/pipeline.png}
\caption{Implemented workflow. Every annotator and crop of one physical observation follows the same fold. Thresholds and instance reconstruction are chosen on calibration before holdout evaluation.}
\Description{Official data passes through grouped splitting, U-Net foreground learning, probability inference, disjoint instance reconstruction, and evaluated COCO output.}
\end{figure*}

\subsection{Inference and instance reconstruction}
INFERENCETEXT
Probabilities are bilinearly restored to $2048\times2048$ before thresholding. Calibration selected threshold THRESHOLD, a disk-shaped binary closing radius of CLOSING native pixels, and minimum component area AREA native pixels. Eight-connected foreground components become instances. This guarantees exclusive pixel ownership but cannot separate touching filaments without an additional instance representation. The area cutoff suppresses spurious fragments at the cost of small-object recall.

Each nonempty component is encoded as compressed COCO RLE using a Fortran-contiguous byte array. Output has precisely two columns, \texttt{filament\_id} and \texttt{segmentation\_rle}, with one row per instance and the original filename stem preserved. A sidecar records coverage of all 180 images, including zero detections. Preflight rejects invalid probabilities, empty masks, duplicates, overlaps, unknown IDs, and incomplete coverage.

\section{Evaluation}\label{sec:evaluation}
\subsection{Metric fidelity and candidate selection}
We reproduce the released organizer implementation of PQ~\cite{panoptic}. For each physical image, one prediction set is compared with each annotator separately. Pairs with strictly greater than 0.5 IoU qualify; matched IoUs, TP, FP, and FN are pooled across annotator-image entries:
\[
\mathrm{PQ}=\frac{\sum_{\mathrm{matched}}\mathrm{IoU}}{\mathrm{TP}+0.5\mathrm{FP}+0.5\mathrm{FN}}.
\]
We retain the official all-qualifying-pairs behavior where GT overlaps complicate textbook one-to-one assumptions. Background is excluded. Numerical parity tests compare the efficient COCO-RLE implementation with the organizer's tensor functions, including empty sets and the strict threshold. Additional tests cover polygon conversion, asymmetric RLE round trips, split/merge penalties, temporal grouping, inference tiling, and CSV ownership.

The initial postprocessing grid contained 16 settings. After excess calibration false positives were observed, nine settings with larger area cutoffs were added. The compact full-frame model improved from PQ 0.25096 to 0.27503 under this adaptive search; the classical local-background-deficit reference reached 0.08265. Expanded capacity/context and overlapping-tile inference were then compared on calibration. The selected candidate reached CALPQ. These maxima are optimistically biased by tuning; they are not independent validation estimates. The model, inference mode, thresholds, and split checksum were frozen before opening holdout results.

\begin{table}[t]
\centering
\caption{Frozen candidate on 145 protected physical images. Counts pool independent annotator comparisons. The interval resamples physical images 1,000 times.}
\begin{tabular}{lr}
\toprule
Measure & Value \\
\midrule
PQ & HOLDOUTPQ \\
95\% bootstrap interval & [CILOW, CIHIGH] \\
Segmentation quality (SQ) & SQVALUE \\
Recognition quality (RQ) & RQVALUE \\
TP / FP / FN & TPVALUE / FPVALUE / FNVALUE \\
Fragmented GT / merged predictions & FRAGVALUE / MERGEVALUE \\
\bottomrule
\end{tabular}
\end{table}

\subsection{Uncertainty and failure analysis}
All annotators of one observation stay together during bootstrap resampling. A stricter sensitivity analysis resamples 27 temporal/duplicate groups: 5,000 draws give interval BLOCKLOW--BLOCKHIGH. Neither interval captures every long-range dependence, random-seed effect, or adaptive research decision. Site intervals and overlap distributions accompany the release. Positive-pair IoU/Dice plots exclude zero pairs; FP/FN counts retain misses. Size-stratified recall uses native-area bins below 1,000, 1,000--10,000, and at least 10,000 pixels.

\begin{figure*}[t]
\centering
\includegraphics[width=\textwidth]{../figures/report-example.png}
\caption{Systematically selected median calibration case, cropped around the largest instance of the first supplied annotator. Cyan shows that annotator; pink shows predictions. The full repository gallery also includes worst and best cases selected by physical-image PQ.}
\Description{A grayscale H-alpha crop alongside annotator boundaries and predicted filament boundaries.}
\end{figure*}

\begin{figure*}[t]
\centering
\includegraphics[width=\textwidth]{../figures/holdout-diagnostics.png}
\caption{Protected holdout diagnostics. Left: site-wise pooled PQ and physical-image bootstrap intervals. Middle: positive-pair overlap distributions; missing objects remain in FP/FN counts. Right: GT recall by native mask area, showing size-dependent misses.}
\Description{Six site-specific PQ estimates with confidence intervals, IoU and Dice distributions, and recall in three instance-size bins.}
\end{figure*}

Our principal limitations are missing faint structures, false positives on other dark features, boundary mismatch, and imperfect connected-component identity. Only 12 of 37 large annotated masks are matched, supporting targeted work on truncation and fragmentation. Independent annotators also disagree about extent and completeness. We preserve the official metric despite its penalties for some physically plausible but unannotated detections. Visual review therefore accompanies quantitative validation. A high pixel overlap is insufficient if the output breaks one object into many fragments or joins distinct objects.

After the original holdout was exposed, an exploratory native-resolution screen retained width 24, 384-pixel crops, and 6,000 updates. Calibration PQ was NATIVEFULL for full-image and NATIVETILED for tiled inference, below CALPQ. The smaller physical context per crop is a tradeoff; higher resolution alone did not improve this recipe. This candidate was rejected without further holdout evaluation or submission. Our next priorities combine broad context with local detail, boundary/affinity supervision, and detector-guided refinement. Edge attention~\cite{edgeattnet}, mask-set prediction~\cite{mask2former}, and skeleton losses~\cite{cldice} motivate these unimplemented alternatives. Subsequent-development nested folds are frozen; chronological/site stress tests and near-duplicate auditing remain needed. Published Dice or pairwise mIoU values are not treated as competition PQ.

\subsection{Reproduction and release status}
The public repository is \url{https://github.com/srivatsav-kannan/solar-seg}. Its canonical notebook runs the CLI's modules for the audit, training, calibration, evaluation, inference, and serialization workflow. Source revisions, checkpoints, and outputs are hashed; Kaggle execution uses an isolated pinned environment. Independent local CPU runs reproduced the exact 1,645-instance CSV, covering all 180 test images with two zero-detection observations. Kaggle submission 56050854 completed successfully with public PQ 0.30. Exact cross-architecture bitwise equality is not assumed. Gates bind correctness, frozen selection, holdout, morphology, and replay evidence to the submitted CSV. The associated public Kaggle notebook and original submitted files are linked from the repository.

"""
details = ""
if author.get("affiliation"):
    details += "\\affiliation{\\institution{" + tex_escape(author["affiliation"]) + "}}\n"
if author.get("email"):
    details += "\\email{" + tex_escape(author["email"]) + "}\n"
inference = (
    f"The selected checkpoint is evaluated using {selected['tile']}-pixel overlapping tiles with 25\\% overlap and positive Hann blending weights."
    if selected.get("tile")
    else "The selected checkpoint is evaluated on the complete resized disk."
)
inference += (
    " Four flip predictions are averaged."
    if selected.get("tta")
    else " No test-time augmentation is used."
)
replacements = {
    "AUTHORDETAILS": details,
    "AUTHOR": tex_escape(author["name"]),
    "FIXEDINTRO": intro,
    "HOLDOUTPQ": f"{h['pq']:.4f}",
    "CILOW": f"{h['bootstrap_95ci'][0]:.4f}",
    "CIHIGH": f"{h['bootstrap_95ci'][1]:.4f}",
    "CALPQ": f"{selected['calibration_pq']:.4f}",
    "NATIVEFULL": f"{screen['results']['unet-native-v1']['pq']:.4f}",
    "NATIVETILED": f"{screen['results']['unet-native-v1-tiled']['pq']:.4f}",
    "BLOCKLOW": f"{block_ci[0]:.4f}",
    "BLOCKHIGH": f"{block_ci[1]:.4f}",
    "WIDTH": str(cfg["width"]),
    "CROPSIZE": str(cfg["crop"]),
    "STEPS": f"{cfg['steps']:,}",
    "TRAINSECONDS": f"{cfg['training_seconds']:.1f}",
    "INFERENCETEXT": inference,
    "THRESHOLD": str(h["params"]["threshold"]),
    "CLOSING": str(h["params"]["closing"]),
    "AREA": str(h["params"]["min_area"]),
    "SQVALUE": f"{h['sq']:.4f}",
    "RQVALUE": f"{h['rq']:.4f}",
    "TPVALUE": str(h["tp"]),
    "FPVALUE": str(h["fp"]),
    "FNVALUE": str(h["fn"]),
    "FRAGVALUE": str(h["fragmented_gt"]),
    "MERGEVALUE": str(h["merged_pred"]),
}
for key, value in replacements.items():
    body = body.replace(key, value)
body = body.replace(
    r"\texttt{MAGFiLO\_1.0\_Annotations\_kaggle2026\_train.json}",
    r"\path{MAGFiLO_1.0_Annotations_kaggle2026_train.json}",
)
(out / "main.tex").write_text(
    "\n".join(
        line.rstrip() for line in (prefix + body + "\n\\clearpage\n" + acknowledgment).splitlines()
    )
    + "\n"
)
shutil.copy2(template_dir / "preamble.tex", out / "preamble.tex")
bib = (
    (template_dir / "main.bib").read_text()
    + r"""
@inproceedings{unet,author={Ronneberger, Olaf and Fischer, Philipp and Brox, Thomas},title={U-Net: Convolutional Networks for Biomedical Image Segmentation},booktitle={MICCAI},year={2015},doi={10.1007/978-3-319-24574-4_28}}
@inproceedings{panoptic,author={Kirillov, Alexander and He, Kaiming and Girshick, Ross and Rother, Carsten and Dollar, Piotr},title={Panoptic Segmentation},booktitle={CVPR},year={2019},eprint={1801.00868}}
@misc{edgeattnet,author={Solomon, Victor and Martens, Piet and Liu, Jingyu and Angryk, Rafal},title={EdgeAttNet: Towards Barb-Aware Filament Segmentation},year={2025},eprint={2509.02964},howpublished={arXiv}}
@inproceedings{mask2former,author={Cheng, Bowen and Misra, Ishan and Schwing, Alexander and Kirillov, Alexander and Girdhar, Rohit},title={Masked-attention Mask Transformer for Universal Image Segmentation},booktitle={CVPR},year={2022},eprint={2112.01527}}
@inproceedings{cldice,author={Shit, Suprosanna and others},title={clDice: A Novel Topology-Preserving Loss Function for Tubular Structure Segmentation},booktitle={CVPR},year={2021},eprint={2003.07311}}
"""
)
bib = bib.replace("@INPROCEEDINGS{ahmadzadeh2024dataset", "@article{ahmadzadeh2024dataset")
bib = bib.replace("booktitle = {Nature Scientific Data}", "journal = {Scientific Data}")
bib = bib.replace("series = {Nature Scientific Data},", "")
bib = bib.replace("eid = {1031},", "")
bib = bib.replace("month = dec,", "month = sep,")  # Publisher-deposited DOI date: 2024-09-27.
(out / "main.bib").write_text("\n".join(line.rstrip() for line in bib.splitlines()) + "\n")
im = Image.open(root / "reports/figures/calibration-examples.png")
im.crop((0, im.height // 3, im.width, 2 * im.height // 3)).save(
    root / "reports/figures/report-example.png"
)
latex = shutil.which("pdflatex") or "/Library/TeX/texbin/pdflatex"
bibtex = shutil.which("bibtex") or "/Library/TeX/texbin/bibtex"
commands = [
    [latex, "-interaction=nonstopmode", "-halt-on-error", "main.tex"],
    [bibtex, "main"],
    [latex, "-interaction=nonstopmode", "-halt-on-error", "main.tex"],
    [latex, "-interaction=nonstopmode", "-halt-on-error", "main.tex"],
]
for command in commands:
    result = subprocess.run(command, cwd=out, text=True, capture_output=True, check=False)
    (build / (Path(command[0]).name + ".log")).write_text(result.stdout + result.stderr)
    if result.returncode:
        raise RuntimeError(result.stdout[-4000:])
shutil.copy2(out / "main.pdf", pdf_out / "baseline-report.pdf")
for path in out.glob("main.*"):
    if path.suffix not in {".tex", ".bib"}:
        shutil.move(path, build / path.name)
print(pdf_out / "baseline-report.pdf")
