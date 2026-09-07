"""Warm-start identity and learnability of the added context branch."""

import pytest
import torch

from research.context import ContextUNet
from research.losses import batch_dice_loss
from research.models import DeepUNet


def test_context_starts_as_exact_parent_and_projection_gets_gradient():
    torch.set_num_threads(4)
    torch.manual_seed(2026)
    parent = DeepUNet(width=4, depth=3).eval()
    candidate = ContextUNet(width=4, depth=3).eval()
    candidate.initialize_base(parent.state_dict())
    x = torch.randn(2, 1, 65, 73)
    expected = parent(x).detach()
    actual = candidate(x)
    assert actual.shape == x.shape
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    target = torch.zeros_like(actual)
    target[:, :, 20:45, 30:34] = 1
    loss = batch_dice_loss(actual, target)
    loss.backward()
    gradient = candidate.context[-1].weight.grad
    assert torch.isfinite(loss) and torch.isfinite(gradient).all()
    assert gradient.abs().sum() > 0


def test_context_rejects_incomplete_or_context_parent():
    model = ContextUNet(width=4)
    with pytest.raises(ValueError, match="exactly"):
        model.initialize_base({})
    with pytest.raises(ValueError, match="exactly"):
        model.initialize_base(model.state_dict())
