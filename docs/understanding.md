# Explain the solution in plain language

The model looks at one grayscale photograph of the Sun. It estimates how likely each pixel is to belong to a filament. We convert those probabilities into separate shapes and encode each shape into a compact string for Kaggle.

PQ measures both object identity and shape. If we paint the right dark pixels but split one filament into ten pieces, we have not identified ten real objects. PQ penalizes missed objects and extra objects as well as poor boundaries. Dice alone cannot fully measure that.

For one correctly detected filament with IoU 0.8, PQ is 0.8. Add one false detection and PQ falls to `0.8 / (1 + 0.5) = 0.5333`. Split a perfect object into two equal halves and neither half exceeds the strict IoU 0.5 matching threshold: PQ becomes zero even though their combined foreground is perfect. This explains why instance reconstruction matters.

Two people may annotate the same photograph. Putting one annotation in training and the other in validation would test whether the model remembers that photograph. Nearby observations can also show the same evolving structure. Our groups and time gap reduce this problem.

Different annotators disagree at faint boundaries and may miss an object. We average their binary foreground masks for training. A pixel marked by one of two annotators receives target 0.5. At evaluation their individual instance annotations stay separate, matching the organizer's evaluator.

U-Net's encoder reduces spatial resolution to learn context. The decoder restores resolution. Skip connections bring back fine detail from earlier layers. The baseline uses three reductions, convolutional blocks, GroupNorm, and SiLU. It has no pretrained filament knowledge.

Binary cross-entropy rewards correct pixel probabilities. Soft Dice balances overlap when foreground is sparse. This teaches foreground, but not object identity directly; that is a baseline limitation.

Thresholding selects pixels; optional small closing fills short gaps; connected components define instances; a calibrated minimum area removes small detections. Every pixel has one owner. Closing can join unrelated filaments and area filtering can discard real small ones, so both must be justified with evidence.

The full public MAGFiLO archive contains the hidden competition answers. Using them directly or through a pretrained model invalidates the experiment. Only the official training labels are used here.

The notebook tells the complete reproduction story. Modules keep the implementation readable and reusable. Saved weights, configuration, input hashes, and output hashes connect the report to the exact submitted predictions.

Before prize review, the participant should be able to explain the split, normalization, augmentation ranges, loss, learning-rate schedule, label handling, instance formation, evaluator, chosen parameters, failure cases, compute requirements, and how to reproduce the CSV. AI assistance does not replace this understanding.
