"""Five cross-validation folds that do not leak.

* All records of a frame go to the same fold, so a second annotator's drawing of a validation
  frame is never seen in training.
* A filament lives for days and the Sun turns once in about 27 days, so frames taken a few days
  apart show the same filaments. Whole 27-day blocks go to one fold.
* The folds are stratified by GONG site, which keeps the image quality of the folds comparable.

The folds I used are frozen in splits/folds.csv; `make_folds` rebuilds them from scratch.
"""
from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


def make_folds(stems: list[str], n_folds: int = 5, block_days: int = 27, seed: int = 2026) -> np.ndarray:
    start = datetime(2010, 1, 1)
    blocks = [(datetime.strptime(s[:14], "%Y%m%d%H%M%S") - start).days // block_days for s in stems]
    sites = [s[-2:] for s in stems]
    fold = np.full(len(stems), -1, int)
    split = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for k, (_, va) in enumerate(split.split(stems, sites, blocks)):
        fold[va] = k
    return fold


def load_folds(cfg: dict) -> dict[str, int]:
    """Frame name -> fold, from splits/folds.csv."""
    df = pd.read_csv(cfg["paths"]["folds"])
    return dict(zip(df["frame"], df["fold"].astype(int)))
