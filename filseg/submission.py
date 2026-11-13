"""Writing the submission and checking it before upload.

One row per filament: `filament_id` is <frame>_<k> and `segmentation_rle` the compressed RLE
counts at 2048 x 2048, without size and without quotes. A frame without filaments has no rows.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pandas as pd
from pycocotools import mask as mask_util

from .rle import NATIVE, to_rle


def rows_for_frame(stem: str, rles: list[dict]) -> list[tuple[str, str]]:
    return [(f"{stem}_{k}", r["counts"].decode("ascii") if isinstance(r["counts"], bytes) else r["counts"])
            for k, r in enumerate(rles, 1)]


def write(rows: list[tuple[str, str]], path: str | Path) -> pd.DataFrame:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows, columns=["filament_id", "segmentation_rle"])
    df.to_csv(path, index=False, quoting=csv.QUOTE_NONE, escapechar=None)      # the host wants the counts unquoted
    return df


def check(df: pd.DataFrame, test_stems: list[str]) -> dict:
    """Everything the host rejects: wrong columns, repeated ids, unknown frames, quotes, empty or overlapping masks."""
    assert list(df.columns) == ["filament_id", "segmentation_rle"], "wrong columns"
    assert df["filament_id"].is_unique, "repeated filament_id"
    stems = df["filament_id"].str.rsplit("_", n=1).str[0]
    assert stems.isin(set(test_stems)).all(), "filament_id of a frame that is not in the test set"
    assert not df["segmentation_rle"].str.contains("[\"']", regex=True).any(), "quotes inside an RLE string"
    areas = []
    for stem, rows in df.groupby(stems):
        rles = [to_rle(c) for c in rows["segmentation_rle"]]
        a = mask_util.area(rles)
        assert (a > 0).all(), f"empty mask in {stem}"
        if len(rles) > 1:
            overlap = np.asarray(mask_util.iou(rles, rles, [0] * len(rles)))
            np.fill_diagonal(overlap, 0)
            assert (overlap == 0).all(), f"overlapping masks in {stem}"
        areas += a.tolist()
    if len(df):
        assert mask_util.decode(to_rle(df["segmentation_rle"].iloc[0])).shape == (NATIVE, NATIVE)
    per_frame = stems.value_counts()
    return {"filaments": int(len(df)), "frames_with_filaments": int(per_frame.size),
            "frames_without": int(len(set(test_stems)) - per_frame.size),
            "median_area_px": float(np.median(areas)) if areas else 0.0}
