"""The whole test-time pipeline for one frame."""
from __future__ import annotations

import json

import numpy as np

from .candidates import Candidate, extract
from .data import solar_disk
from .inference import on_disk, probability_map
from .model import load_net
from .scorers import KeepRule


class Segmenter:
    """Loads the network and both scorers from weights/ and turns a frame into its filaments."""

    def __init__(self, cfg: dict, device: str = "cuda"):
        weights = cfg["paths"]["weights"]
        self.post = json.loads((weights / "postprocess.json").read_text())
        self.net = load_net(cfg, weights / self.post["network"], device)
        self.rule = KeepRule(weights, self.post)
        self.tta = cfg["predict"]["tta"]

    def __call__(self, img: np.ndarray) -> list[Candidate]:
        disk = solar_disk(img)
        prob = on_disk(probability_map(self.net, img, self.tta), disk)
        cands = extract(prob, img, disk, self.post["threshold"], self.post["min_area"])
        return self.rule.keep(img, prob, cands)
