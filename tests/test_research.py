"""Research models retain image geometry and cannot corrupt a baseline checkpoint."""

import numpy as np
import torch

from research.make_scoring_folds import build_scoring_folds
from research.models import DeepUNet
from research.reconstruction import reconstruct
from solarseg.model import UNet, segmentation_loss
from solarseg.postprocess import decode, instances


def test_depth_three_is_exactly_compatible_with_released_architecture():
    torch.manual_seed(1)
    base, deeper = UNet(8), DeepUNet(8, 3)
    deeper.load_state_dict(base.state_dict(), strict=True)
    image = torch.randn(2, 1, 65, 79)
    with torch.no_grad():
        torch.testing.assert_close(base(image), deeper(image), rtol=0, atol=0)


def test_deep_model_preserves_odd_shapes_and_has_finite_gradients():
    model = DeepUNet(8, 4)
    image = torch.randn(2, 1, 65, 79)
    target = torch.zeros_like(image)
    target[:, :, 23:39, 5:59] = 1
    output = model(image)
    assert output.shape == image.shape
    loss = segmentation_loss(output, target)
    loss.backward()
    assert torch.isfinite(loss)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_seeded_reconstruction_recovers_connected_weak_region_and_rejects_unseeded():
    probability = np.zeros((64, 64), dtype=np.float32)
    probability[5:10, 5:14] = 0.9
    probability[5:10, 14:30] = 0.5
    probability[40:50, 40:50] = 0.5
    result = reconstruct(
        probability, low=0.4, high=0.8, min_area=10, closing=0, min_seed_area=8, native_size=64
    )
    assert len(result) == 1
    mask = decode(result[0])
    assert mask[7, 28] and not mask[45, 45]
    assert mask.sum() == 125


def test_equal_seed_threshold_reproduces_connected_component_baseline():
    rng = np.random.default_rng(42)
    probability = rng.uniform(size=(64, 64)).astype(np.float32)
    actual = reconstruct(
        probability, low=0.6, high=0.6, min_area=8, closing=0, min_seed_area=1, native_size=64
    )
    expected = instances(probability, threshold=0.6, min_area=8, closing=0, native_size=64)
    assert [r["counts"] for r in actual] == [r["counts"] for r in expected]


def test_scoring_folds_exclude_outer_images_and_embargo_inner_calibration():
    import pandas as pd

    frame = pd.read_csv("configs/manifests/train_manifest.csv")
    outputs = build_scoring_folds(frame)
    for outer, split in outputs.items():
        assert set(split.loc[split.role == "holdout", "fold"]) == {outer}
        assert set(split.loc[split.role == "calibration", "fold"]) == {1}
        assert split.loc[split.role != "embargo"].groupby("group").role.nunique().max() == 1
        dates = pd.to_datetime(split.date).dt.as_unit("ns").astype("int64").to_numpy()
        for a, b in [("train", "calibration"), ("train", "holdout"), ("calibration", "holdout")]:
            gap = np.abs(
                dates[(split.role == a).to_numpy(), None]
                - dates[(split.role == b).to_numpy()][None, :]
            )
            assert gap.min() > 3 * 86400 * 10**9
