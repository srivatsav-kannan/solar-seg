"""Confidence-seeded reconstruction for weak but connected filament regions."""

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

from solarseg.postprocess import encode_component


def reconstruct(
    probability,
    low=0.45,
    high=0.8,
    min_area=200,
    closing=3,
    min_seed_area=16,
    native_size=2048,
):
    probability = np.asarray(probability)
    if (
        probability.ndim != 2
        or not np.isfinite(probability).all()
        or probability.min() < 0
        or probability.max() > 1
    ):
        raise ValueError("Expected a finite 2D probability map in [0, 1]")
    if not 0 < low <= high < 1 or min_area < 1 or min_seed_area < 1:
        raise ValueError("Invalid probability/area parameters")
    if closing < 0 or int(closing) != closing:
        raise ValueError("Closing radius must be a nonnegative integer")
    prob = np.asarray(
        Image.fromarray(probability.astype(np.float32)).resize(
            (native_size, native_size), Image.Resampling.BILINEAR
        )
    )
    binary = prob >= low
    if closing:
        y, x = np.ogrid[-closing : closing + 1, -closing : closing + 1]
        binary = ndi.binary_closing(binary, structure=x * x + y * y <= closing**2)
    labels, count = ndi.label(binary, structure=np.ones((3, 3)))
    areas = np.bincount(labels.ravel(), minlength=count + 1)
    seeds = np.bincount(labels[prob >= high], minlength=count + 1)
    bounds = ndi.find_objects(labels)
    retained = np.flatnonzero((areas >= min_area) & (seeds >= min_seed_area))
    return [encode_component(labels, int(i), bounds[i - 1]) for i in retained if i > 0]
