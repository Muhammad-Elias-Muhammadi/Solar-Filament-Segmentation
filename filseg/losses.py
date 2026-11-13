"""Loss for soft, sparse and thin targets.

* Binary cross-entropy against the soft target keeps the output a calibrated probability, which
  the threshold and the scorers rely on.
* Soft Dice per crop counters the imbalance: filaments cover well under one percent of a frame.
* The centre-line term (cross-entropy on the auxiliary head plus skeleton recall on the main one,
  Kirchhoff et al. 2024) punishes a filament broken in two, which PQ punishes twice: one missed
  filament and two false ones.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def soft_dice(logits: torch.Tensor, target: torch.Tensor, eps: float = 1.0) -> torch.Tensor:
    p, t = torch.sigmoid(logits).flatten(1), target.flatten(1)
    return (1 - (2 * (p * t).sum(1) + eps) / (p.sum(1) + t.sum(1) + eps)).mean()


def skeleton_recall(logits: torch.Tensor, skel: torch.Tensor, eps: float = 1.0) -> torch.Tensor:
    """One minus the share of the target centre line that the prediction covers."""
    p, s = torch.sigmoid(logits).flatten(1), skel.flatten(1)
    return (1 - ((p * s).sum(1) + eps) / (s.sum(1) + eps)).mean()


class FilamentLoss(torch.nn.Module):
    def __init__(self, bce_weight: float = 1.0, dice_weight: float = 0.5, skel_weight: float = 0.2):
        super().__init__()
        self.w = (bce_weight, dice_weight, skel_weight)

    def forward(self, logits: torch.Tensor, soft: torch.Tensor, skel: torch.Tensor) -> torch.Tensor:
        logits = logits.float()
        fg, centre = logits[:, :1], logits[:, 1:2]
        bce = F.binary_cross_entropy_with_logits(fg, soft)
        line = F.binary_cross_entropy_with_logits(centre, skel) + skeleton_recall(fg, skel)
        return self.w[0] * bce + self.w[1] * soft_dice(fg, soft) + self.w[2] * line
