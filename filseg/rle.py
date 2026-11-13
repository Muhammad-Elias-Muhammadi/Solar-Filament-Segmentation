"""Run-length encoding in the format of the competition.

The submission stores only the compressed COCO counts string of each mask, without the size
(always 2048 x 2048) and without quotes. pycocotools wants Fortran-ordered uint8 arrays; a C-ordered
array is silently read as the transposed mask.
"""
from __future__ import annotations

import numpy as np
from pycocotools import mask as mask_util

NATIVE = 2048


def encode(mask: np.ndarray) -> str:
    return mask_util.encode(np.asfortranarray(mask.astype(np.uint8)))["counts"].decode("ascii")


def to_rle(counts: str | bytes) -> dict:
    return {"size": [NATIVE, NATIVE], "counts": counts.encode("ascii") if isinstance(counts, str) else counts}


def decode(counts: str) -> np.ndarray:
    return mask_util.decode(to_rle(counts)).astype(bool)


def from_polygons(polygons: list[list[float]]) -> dict:
    """One filament of the annotation file (a list of polygons) as an RLE object."""
    rles = mask_util.frPyObjects(polygons, NATIVE, NATIVE)
    return mask_util.merge(rles) if isinstance(rles, list) else rles
