"""Settings and paths. Everything is read from config.yaml at the root of the repository."""
from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_config(path: str | Path | None = None) -> dict:
    with open(Path(path) if path else ROOT / "config.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["paths"] = _paths(cfg)
    return cfg


def _paths(cfg: dict) -> dict[str, Path]:
    def here(p: str) -> Path:
        p = Path(p)
        return p if p.is_absolute() else ROOT / p

    data, work = here(cfg["data_dir"]), here(cfg["work_dir"])
    return {
        "train_images": data / "train" / "train_images",
        "test_images": data / "test" / "test_images",
        "annotations": data / "train" / "MAGFiLO_1.0_Annotations_kaggle2026_train.json",
        "cache": work / "cache",
        "runs": work / "runs",
        "tuning": work / "tuning",
        "weights": here(cfg["weights_dir"]),
        "folds": ROOT / "splits" / "folds.csv",
        "submission": ROOT / "submission" / "submission.csv",
    }


def seed_everything(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
