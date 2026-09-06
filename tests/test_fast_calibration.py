"""Area-cutoff reuse must reproduce every original per-annotator count and diagnostic."""

import numpy as np

from research.fast_calibration import image_grid
from solarseg.metrics import evaluate_image
from solarseg.postprocess import encode, instances


def test_reused_components_match_original_masks_and_all_evaluation_rows():
    size = 64
    probability = np.zeros((32, 32), np.float32)
    probability[1:5, 0:8] = 0.6
    probability[12:25, 10:25] = 0.9
    probability[17:19, 13:15] = 0.0
    probability[28:30, 28:30] = 0.8
    a = np.zeros((size, size), np.uint8)
    a[24:50, 20:50] = 1
    b = a.copy()
    b[20:40, 10:30] = 1

    class Data:
        def ground_truth(self, stem):
            return {"annotator-one": [encode(a), encode(b)], "annotator-two": []}

    data = Data()
    grid = [
        {"threshold": t, "min_area": area, "closing": closing}
        for t in [0.6, 0.8, 0.9]
        for area in [4, 16, 128, 900]
        for closing in [0, 2]
    ]
    actual = image_grid(data, "example", probability, grid, native_size=size)
    expected = [
        evaluate_image(data, "example", instances(probability, native_size=size, **p)) for p in grid
    ]
    assert actual == expected
