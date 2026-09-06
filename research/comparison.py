"""Paired pooled-PQ comparisons with temporal groups kept together."""

from collections import defaultdict

import numpy as np

from solarseg.metrics import aggregate


def paired_comparison(reference, candidate, stem_groups, replicates=5000, seed=2026):
    def indexed(rows):
        result = {(r["stem"], r["annotator_image"]): r for r in rows}
        if len(result) != len(rows):
            raise ValueError("Repeated annotator-image record")
        return result

    baseline, challenger = indexed(reference), indexed(candidate)
    if not baseline or baseline.keys() != challenger.keys():
        raise ValueError("Paired comparison requires exactly the same observations and annotators")
    groups = defaultdict(lambda: np.zeros((2, 4), dtype=np.float64))
    columns = ["iou_sum", "tp", "fp", "fn"]
    for key in sorted(baseline):
        a, b = baseline[key], challenger[key]
        if a.get("n_gt") != b.get("n_gt"):
            raise ValueError("Ground-truth instance counts differ")
        group = stem_groups[key[0]]
        groups[group][0] += [a[c] for c in columns]
        groups[group][1] += [b[c] for c in columns]
    values = np.stack(list(groups.values()))

    def pq(x):
        den = x[..., 1] + 0.5 * (x[..., 2] + x[..., 3])
        return np.divide(x[..., 0], den, out=np.zeros_like(den), where=den > 0)

    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(replicates, len(values)))
    scores = pq(values[indices].sum(axis=1))
    difference = scores[:, 1] - scores[:, 0]
    a, b = aggregate(reference), aggregate(candidate)
    return {
        "reference": a,
        "candidate": b,
        "pq_gain": b["pq"] - a["pq"],
        "paired_block_bootstrap_95ci": np.quantile(difference, [0.025, 0.975]).tolist(),
        "bootstrap_replicates": replicates,
        "seed": seed,
        "groups": len(groups),
        "physical_images": len({r["stem"] for r in reference}),
        "annotator_records": len(reference),
        "limitation": "Fixed-model group resampling; this does not remove model-selection bias or capture retraining uncertainty.",
    }
