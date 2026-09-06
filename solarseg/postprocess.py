"""Deterministic native-resolution instances with disjoint pixel ownership."""

import numpy as np
from PIL import Image
from pycocotools import mask as coco_mask
from scipy import ndimage as ndi


def encode(mask):
    return coco_mask.encode(np.asfortranarray(mask, dtype=np.uint8))


def decode(rle):
    return coco_mask.decode(rle).astype(bool)


def encode_component(labels, label, bounds):
    """Encode sparse components without allocating one full-disk mask per object."""
    ys, xs = bounds
    columns, rows = np.nonzero((labels[bounds] == label).T)
    indices = rows + ys.start + (columns + xs.start) * labels.shape[0]
    breaks = np.flatnonzero(np.diff(indices) != 1)
    starts = np.concatenate(([indices[0]], indices[breaks + 1]))
    ends = np.concatenate((indices[breaks], [indices[-1]]))
    counts = [int(starts[0])]
    for i, (start, end) in enumerate(zip(starts, ends, strict=True)):
        if i:
            counts.append(int(start - ends[i - 1] - 1))
        counts.append(int(end - start + 1))
    trailing = int(labels.size - ends[-1] - 1)
    if trailing:
        counts.append(trailing)
    return coco_mask.frPyObjects({"size": list(labels.shape), "counts": counts}, *labels.shape)


def instances(probability, threshold=0.5, min_area=100, closing=0, native_size=2048):
    probability = np.asarray(probability)
    if (
        probability.ndim != 2
        or not np.isfinite(probability).all()
        or probability.min() < 0
        or probability.max() > 1
    ):
        raise ValueError("Expected a finite 2D probability map in [0, 1]")
    if not 0 < threshold < 1 or min_area < 1 or closing < 0 or int(closing) != closing:
        raise ValueError("Invalid instance reconstruction parameters")
    prob = np.asarray(
        Image.fromarray(probability.astype(np.float32)).resize(
            (native_size, native_size), Image.Resampling.BILINEAR
        )
    )
    binary = prob >= threshold
    if closing:
        y, x = np.ogrid[-closing : closing + 1, -closing : closing + 1]
        binary = ndi.binary_closing(binary, structure=x * x + y * y <= closing * closing)
    labels, _n = ndi.label(binary, structure=np.ones((3, 3)))
    areas = np.bincount(labels.ravel())
    bounds = ndi.find_objects(labels)
    result = []
    for label in np.flatnonzero(areas[1:] >= min_area) + 1:
        result.append(encode_component(labels, int(label), bounds[label - 1]))
    return result
