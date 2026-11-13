"""Candidate filaments: the connected components of the thresholded probability map.

Every 8-connected component above the threshold is one candidate, so the masks never overlap,
which the leaderboard requires. For each candidate I also compute twenty numbers that the feature
scorer uses to judge it: size and shape, how sure the network is inside it, how dark it is against
a ring around it, where it lies on the disk, and how many candidates the frame has.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import torch
from pycocotools import mask as mask_util

from .data import disk_mask
from .rle import NATIVE

FEATURES = [
    "area", "log_area", "bw", "bh", "fill", "elong", "p_mean", "p_max", "p_std",
    "f50", "f70", "f90", "inten_in", "inten_ring", "contrast", "radius", "n_comp_img",
    "member_std", "pieces_hi", "thr",
]
RING = np.ones((15, 15), np.uint8)       # the ring a candidate's darkness is compared against
PAD = 8


@dataclass
class Candidate:
    mask_rle: dict        # pycocotools RLE at 2048 x 2048
    feats: dict
    bbox: tuple           # x, y, w, h


def _elongation(m: np.ndarray) -> float:
    """Ratio of the long to the short axis of the mask's second moments."""
    mo = cv2.moments(m, binaryImage=True)
    if mo["m00"] == 0:
        return 0.0
    a, b, c = mo["mu20"] / mo["m00"], mo["mu11"] / mo["m00"], mo["mu02"] / mo["m00"]
    d = np.sqrt(max(0.0, ((a - c) / 2) ** 2 + b ** 2))
    return float(np.sqrt(((a + c) / 2 + d) / max((a + c) / 2 - d, 1e-6)))


def extract(prob: torch.Tensor, img: np.ndarray, disk: tuple, threshold: float, min_area: int = 30) -> list[Candidate]:
    """Candidates of one frame. `prob` is the probability map with the sky already set to zero."""
    p = prob.float().cpu().numpy()
    n, lab, stats, _ = cv2.connectedComponentsWithStats((p > threshold).astype(np.uint8), connectivity=8)
    cx, cy, radius = disk
    on = img[disk_mask(disk)]
    norm = float(np.median(on)) + 1e-3 if on.size else 1.0
    out = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < max(1, min_area):
            continue
        x0, y0 = max(0, x - PAD), max(0, y - PAD)
        x1, y1 = min(NATIVE, x + w + PAD), min(NATIVE, y + h + PAD)
        m = lab[y0:y1, x0:x1] == i
        pv = p[y0:y1, x0:x1][m]
        m8 = m.astype(np.uint8)
        ring = cv2.dilate(m8, RING).astype(bool) & ~m
        iv = img[y0:y1, x0:x1]
        inten_in = float(iv[m].mean()) / norm
        inten_ring = float(iv[ring].mean()) / norm if ring.any() else inten_in
        core = (p[y0:y1, x0:x1] > min(0.95, threshold + 0.2)) & m
        feats = {
            "area": float(area), "log_area": float(np.log1p(area)), "bw": float(w), "bh": float(h),
            "fill": float(area) / (w * h), "elong": _elongation(m8),
            "p_mean": float(pv.mean()), "p_max": float(pv.max()), "p_std": float(pv.std()),
            "f50": float((pv > 0.5).mean()), "f70": float((pv > 0.7).mean()), "f90": float((pv > 0.9).mean()),
            "inten_in": inten_in, "inten_ring": inten_ring, "contrast": inten_ring - inten_in,
            "radius": float(np.hypot(x + w / 2 - cx, y + h / 2 - cy) / max(radius, 1)),
            "n_comp_img": 0.0,
            "member_std": 0.0,        # spread between networks of an ensemble; zero with one network
            "pieces_hi": float(cv2.connectedComponents(core.astype(np.uint8), connectivity=8)[0] - 1),
            "thr": float(threshold),
        }
        full = np.zeros((NATIVE, NATIVE), np.uint8, order="F")
        full[y0:y1, x0:x1] = m8
        out.append(Candidate(mask_util.encode(full), feats, (x, y, w, h)))
    for c in out:
        c.feats["n_comp_img"] = float(len(out))
    return out


def feature_matrix(cands: list[Candidate]) -> np.ndarray:
    return np.array([[c.feats[f] for f in FEATURES] for c in cands], np.float32).reshape(-1, len(FEATURES))
