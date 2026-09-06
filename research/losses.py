"""A batch-Dice ablation for foreground-sparse crops; no change to the released loss."""

from torch.nn import functional as F


def batch_dice_loss(logits, target):
    probability = logits.sigmoid()
    dice = 1 - (2 * (probability * target).sum() + 1) / (probability.sum() + target.sum() + 1)
    return F.binary_cross_entropy_with_logits(logits, target) + dice
