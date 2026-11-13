"""The two keep-scorers and the keep rule.

Adding a candidate to a frame with n annotator records adds its matched IoU to the numerator of
pooled PQ for every record it matches (IoU > 0.5), and exactly n/2 to the denominator. So keeping
it raises PQ exactly when

    y = (1/n) * sum over records of IoU * [IoU > 0.5]

is larger than PQ/2. Both scorers regress this y, and a candidate is kept when the mean of their
two estimates is above tau (0.25, close to half the PQ I reach).

* The feature scorer is a gradient-boosted regressor on the twenty numbers of candidates.py.
* The patch scorer is a ResNet-18 that looks at the candidate itself: a 128 px patch with six
  channels, the image, the probability map and the candidate mask, once in a window of 1.5 times
  its bounding box and once in a window of 4 times for context. The patch is scaled to a fixed
  size, so it loses the absolute size of the candidate; the feature scorer keeps it. That is one
  reason the mean of the two beats either alone.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import timm
import torch
import torch.nn.functional as F
from pycocotools import mask as mask_util
from safetensors.torch import load_file, save_file
from sklearn.ensemble import HistGradientBoostingRegressor

from .augment import dihedral
from .candidates import FEATURES, feature_matrix

PATCH = 128
VIEWS = ((1.5, 64), (4.0, 256))        # (window as a multiple of the bounding box, smallest window in px)
CHANNELS = 3 * len(VIEWS)
IMAGE_CHANNELS = [0, 3]                # channels that get the brightness jitter in training


# ------------------------------------------------------------------------------ feature scorer
def feature_scorer(seed: int = 0) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(max_iter=400, learning_rate=0.04, max_leaf_nodes=31,
                                         min_samples_leaf=40, l2_regularization=1.0, random_state=seed)


def feature_scores_out_of_fold(table: pd.DataFrame, folds: np.ndarray, seed: int = 0) -> np.ndarray:
    """Each candidate scored by a regressor fitted on the other folds."""
    X, y, out = table[FEATURES].to_numpy(np.float32), table["y"].to_numpy(), np.zeros(len(table))
    for k in np.unique(folds):
        out[folds == k] = feature_scorer(seed).fit(X[folds != k], y[folds != k]).predict(X[folds == k])
    return out


def save_pickle(model, path: str | Path) -> None:
    with open(path, "wb") as f:
        pickle.dump(model, f)


def load_pickle(path: str | Path):
    with open(path, "rb") as f:
        return pickle.load(f)


# -------------------------------------------------------------------------------- patch scorer
def _window(arr: np.ndarray, cx: float, cy: float, side: int) -> np.ndarray:
    """Square window of `side` px around (cx, cy), zero outside the frame, resized to PATCH."""
    x0, y0 = int(round(cx - side / 2)), int(round(cy - side / 2))
    out = np.zeros((side, side), arr.dtype)
    sx0, sy0 = max(x0, 0), max(y0, 0)
    sx1, sy1 = min(x0 + side, arr.shape[1]), min(y0 + side, arr.shape[0])
    if sx1 > sx0 and sy1 > sy0:
        out[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = arr[sy0:sy1, sx0:sx1]
    return cv2.resize(out, (PATCH, PATCH), interpolation=cv2.INTER_AREA if side > PATCH else cv2.INTER_LINEAR)


def patches(img: np.ndarray, prob8: np.ndarray, cands: list) -> np.ndarray:
    """(n, 6, 128, 128) uint8 for the candidates of one frame; `prob8` is the map times 255."""
    out = np.zeros((len(cands), CHANNELS, PATCH, PATCH), np.uint8)
    for j, c in enumerate(cands):
        x, y, w, h = c.bbox
        mask = mask_util.decode(c.mask_rle) * 255
        for v, (mult, smallest) in enumerate(VIEWS):
            side = int(max(mult * max(w, h), smallest))
            for k, arr in enumerate((img, prob8, mask)):
                out[j, 3 * v + k] = _window(arr, x + w / 2, y + h / 2, side)
    return out


def patch_net(arch: str, pretrained: bool) -> torch.nn.Module:
    net = timm.create_model(arch, pretrained=pretrained, in_chans=CHANNELS, num_classes=1, drop_rate=0.2)
    return net.to(memory_format=torch.channels_last)


def fit_patch_scorer(X: torch.Tensor, y: torch.Tensor, rows: np.ndarray, cfg: dict, seed: int) -> torch.nn.Module:
    """One network trained on the candidates `rows`. X is uint8 on the GPU, y the soft target."""
    torch.manual_seed(seed)
    net = patch_net(cfg["arch"], pretrained=True).cuda()
    batch = cfg["batch"]
    opt = torch.optim.AdamW(net.parameters(), lr=cfg["lr"], weight_decay=0.05)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, cfg["lr"], total_steps=cfg["epochs"] * (len(rows) // batch), pct_start=0.1)
    rows = torch.as_tensor(rows, device="cuda")
    net.train()
    for _ in range(cfg["epochs"]):
        order = rows[torch.randperm(len(rows), device="cuda")]
        for i in range(0, len(order) - batch + 1, batch):
            idx = order[i:i + batch]
            xb = dihedral(X[idx].float().div(255), int(torch.randint(8, (1,))))
            gain = 0.9 + 0.2 * torch.rand(len(idx), 1, 1, 1, device="cuda")
            xb[:, IMAGE_CHANNELS] = (xb[:, IMAGE_CHANNELS] * gain).clamp(0, 1)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logit = net(xb.contiguous(memory_format=torch.channels_last)).float()[:, 0]
            loss = F.binary_cross_entropy_with_logits(logit, y[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
    return net.eval()


@torch.inference_mode()
def patch_scores(nets: list[torch.nn.Module], X, rows: np.ndarray | None = None, chunk: int = 128) -> np.ndarray:
    """Mean over the networks and the eight flips and rotations of each patch."""
    X = torch.as_tensor(X, device="cuda")
    rows = np.arange(len(X)) if rows is None else rows
    out = np.zeros(len(rows), np.float32)
    for i in range(0, len(rows), chunk):
        xb = X[torch.as_tensor(rows[i:i + chunk], device="cuda")].float().div(255)
        total = torch.zeros(len(xb), device="cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            views = torch.cat([dihedral(xb, k) for k in range(8)]).contiguous(memory_format=torch.channels_last)
            for net in nets:
                total += torch.sigmoid(net(views).float()[:, 0]).view(8, len(xb)).sum(0)
        out[i:i + chunk] = (total / (8 * len(nets))).cpu().numpy()
    return out


def save_patch_scorer(net: torch.nn.Module, path: str | Path) -> None:
    state = {k: (v.half() if v.is_floating_point() else v).contiguous().cpu() for k, v in net.state_dict().items()}
    save_file(state, str(path))


def load_patch_scorer(arch: str, path: str | Path) -> torch.nn.Module:
    net = patch_net(arch, pretrained=False)
    net.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in load_file(str(path)).items()})
    return net.cuda().eval()


# --------------------------------------------------------------------------------- keep rule
class KeepRule:
    """Loads both scorers from the weights folder and decides which candidates of a frame to keep."""

    def __init__(self, weights: Path, post: dict):
        self.features = load_pickle(weights / post["feature_scorer"])
        self.patch_arch = post["patch_scorer"]["arch"]
        self.patch_nets = [load_patch_scorer(self.patch_arch, weights / f) for f in post["patch_scorer"]["files"]]
        self.blend, self.tau = post["blend"], post["keep_tau"]

    def scores(self, img: np.ndarray, prob: torch.Tensor, cands: list) -> np.ndarray:
        if not cands:
            return np.zeros(0)
        prob8 = (prob.cpu().numpy() * 255).round().astype(np.uint8)
        a = self.features.predict(feature_matrix(cands))
        b = patch_scores(self.patch_nets, patches(img, prob8, cands))
        return self.blend * a + (1 - self.blend) * b

    def keep(self, img: np.ndarray, prob: torch.Tensor, cands: list) -> list:
        return [c for c, s in zip(cands, self.scores(img, prob, cands)) if s > self.tau]
