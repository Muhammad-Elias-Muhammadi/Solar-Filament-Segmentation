"""The probability map of a whole frame.

The network is fully convolutional, so after training on 1024 px crops it runs on the whole
2048 x 2048 frame in one pass. Nothing is resized. The map is the mean over flipped copies of the
frame (test-time augmentation), each flipped back before averaging.
"""
from __future__ import annotations

import numpy as np
import torch

from .augment import dihedral, dihedral_inv
from .data import disk_mask

VIEWS = {1: [0], 2: [0, 1], 4: [0, 1, 2, 3], 8: list(range(8))}


@torch.inference_mode()
def probability_map(net: torch.nn.Module, img: np.ndarray, tta: int = 4, device: str = "cuda") -> torch.Tensor:
    """(2048, 2048) float32 probability on `device` for a uint8 frame."""
    x = torch.from_numpy(np.array(img)).to(device).float().div_(255)[None, None]     # copy: cached frames are read-only
    prob = torch.zeros_like(x)
    for k in VIEWS[tta]:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logit = net(dihedral(x, k).contiguous(memory_format=torch.channels_last))[:, :1]
        prob += dihedral_inv(torch.sigmoid(logit.float()), k)
    return (prob / len(VIEWS[tta]))[0, 0]


def on_disk(prob: torch.Tensor, disk: tuple) -> torch.Tensor:
    """Sets the sky around the solar disk to zero."""
    return prob * torch.from_numpy(disk_mask(disk)).to(prob.device)


def to_uint8(prob: torch.Tensor) -> np.ndarray:
    """Maps are stored as uint8 (times 255): 4 MB per frame instead of 16."""
    return (prob.clamp(0, 1) * 255 + 0.5).to(torch.uint8).cpu().numpy()
