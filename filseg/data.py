"""Reading the frames and annotations, and the training cache.

A frame can carry up to three annotator records. The image id in the annotation file is
<annotator batch>-<frame>, so records of the same frame share the frame name. For scoring they
stay separate (the host scores each record); for training they are averaged into one soft target,
the fraction of annotators that marked a pixel. Only the segmentation polygons are read.

The cache holds every training frame at native resolution, decoded once:

    img2048.npy   (N, 2048, 2048) uint8   the H-alpha frame
    soft2048.npy  (N, 2048, 2048) uint8   fraction of annotators marking the pixel, times 255
    skel2048.npy  (N, 2048, 2048) uint8   centre line of the majority mask, 3 px wide, times 255
    disk.npy      (N, 3) float32          solar disk (cx, cy, r) in pixels
    stems.json                            frame names, in the order of the arrays
    bank.pkl                              every majority filament with some context, for copy-paste
"""
from __future__ import annotations

import json
import pickle
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from pycocotools import mask as mask_util
from skimage.morphology import skeletonize

from .rle import NATIVE, from_polygons

BANK_PAD = 24        # context kept around each filament cut out for copy-paste (px)
BG_KERNEL = 41       # median filter size for the local background of a filament


@dataclass
class Record:
    record_id: str       # "010401-20160920230134Lh"
    stem: str            # "20160920230134Lh"
    polygons: list       # one list of polygons per filament


def load_records(cfg: dict) -> dict[str, list[Record]]:
    """Frame name -> its annotator records, sorted by frame name."""
    with open(cfg["paths"]["annotations"], encoding="utf-8") as f:
        coco = json.load(f)
    polygons = defaultdict(list)
    for ann in coco["annotations"]:
        polygons[ann["image_id"]].append(ann["segmentation"])
    out = defaultdict(list)
    for img in coco["images"]:
        rid = img["id"]
        stem = rid.split("-", 1)[1]
        out[stem].append(Record(rid, stem, polygons.get(rid, [])))
    return dict(sorted(out.items()))


def read_image(path: str | Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise IOError(f"cannot read {path}")
    if img.shape != (NATIVE, NATIVE):
        raise ValueError(f"{path}: expected {NATIVE}x{NATIVE}, got {img.shape}")
    return img


def solar_disk(img: np.ndarray) -> tuple[float, float, float]:
    """Centre and radius of the solar disk in pixels. Everything outside it is sky."""
    small = cv2.GaussianBlur(cv2.resize(img, (512, 512), interpolation=cv2.INTER_AREA), (5, 5), 0)
    thr = max(10.0, 0.25 * float(np.percentile(small, 99)))
    contours, _ = cv2.findContours((small > thr).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return NATIVE / 2, NATIVE / 2, NATIVE / 2
    (cx, cy), r = cv2.minEnclosingCircle(max(contours, key=cv2.contourArea))
    s = NATIVE / 512
    return cx * s, cy * s, r * s


def disk_mask(disk: tuple[float, float, float], margin: float = 1.01) -> np.ndarray:
    yy, xx = np.ogrid[:NATIVE, :NATIVE]
    cx, cy, r = disk[0], disk[1], disk[2] * margin
    return (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r


def record_mask(rec: Record) -> np.ndarray:
    m = np.zeros((NATIVE, NATIVE), np.uint8)
    for poly in rec.polygons:
        m |= mask_util.decode(from_polygons(poly))
    return m


# ---------------------------------------------------------------------------- training cache
def _frame(args) -> dict:
    path, records = args
    img = read_image(path)
    soft = np.mean([record_mask(r).astype(np.float32) for r in records], axis=0)
    major = (soft >= 0.5).astype(np.uint8)
    skel = cv2.dilate(skeletonize(major.astype(bool)).astype(np.uint8), np.ones((3, 3), np.uint8))
    soft8 = np.round(soft * 255).astype(np.uint8)
    return {"img": img, "disk": solar_disk(img), "soft": soft8, "skel": skel * 255,
            "bank": _bank(img, soft8, skel * 255, major)}


def _bank(img: np.ndarray, soft: np.ndarray, skel: np.ndarray, major: np.ndarray) -> list[dict]:
    """Cut every majority filament out of the frame for copy-paste.

    `ratio` is the filament's darkening against its local background (image divided by its median
    filter). Pasting multiplies the destination by this ratio, so a pasted filament takes on the
    brightness of where it lands and keeps its own contrast.
    """
    n, lab, stats, _ = cv2.connectedComponentsWithStats(major, connectivity=8)
    out = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < 40:
            continue
        x0, y0 = max(0, x - BANK_PAD), max(0, y - BANK_PAD)
        x1, y1 = min(NATIVE, x + w + BANK_PAD), min(NATIVE, y + h + BANK_PAD)
        patch = img[y0:y1, x0:x1]
        ratio = np.clip(patch.astype(np.float32) / (cv2.medianBlur(patch, BG_KERNEL).astype(np.float32) + 1.0), 0.2, 1.3)
        own = lab[y0:y1, x0:x1] == i
        out.append({"ratio": ratio.astype(np.float16),
                    "soft": np.where(own, soft[y0:y1, x0:x1], 0).astype(np.uint8),
                    "skel": np.where(own, skel[y0:y1, x0:x1], 0).astype(np.uint8)})
    return out


def build_cache(cfg: dict, workers: int = 12, progress=None) -> Path:
    """Decode and rasterise every training frame once (about six minutes with twelve workers)."""
    out = cfg["paths"]["cache"]
    out.mkdir(parents=True, exist_ok=True)
    records = load_records(cfg)
    stems = list(records)
    jobs = [(cfg["paths"]["train_images"] / f"{s}.jpeg", records[s]) for s in stems]
    arrays = {k: np.lib.format.open_memmap(out / f"{k}2048.npy", mode="w+", dtype=np.uint8,
                                           shape=(len(stems), NATIVE, NATIVE)) for k in ("img", "soft", "skel")}
    disks, bank = np.zeros((len(stems), 3), np.float32), []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        frames = ex.map(_frame, jobs, chunksize=2)
        if progress is not None:
            frames = progress(frames, total=len(jobs), desc="cache")
        for i, r in enumerate(frames):
            for k in arrays:
                arrays[k][i] = r[k]
            disks[i] = r["disk"]
            bank += [{**e, "frame": i} for e in r["bank"]]
    for a in arrays.values():
        a.flush()
    np.save(out / "disk.npy", disks)
    (out / "stems.json").write_text(json.dumps(stems))
    with open(out / "bank.pkl", "wb") as f:
        pickle.dump(bank, f, protocol=pickle.HIGHEST_PROTOCOL)
    return out


def load_cache(cfg: dict) -> dict:
    d = cfg["paths"]["cache"]
    if not (d / "bank.pkl").exists():
        raise FileNotFoundError(f"no cache in {d}; run scripts/build_cache.py first")
    with open(d / "bank.pkl", "rb") as f:
        bank = pickle.load(f)
    return {"stems": json.loads((d / "stems.json").read_text()),
            "img": np.load(d / "img2048.npy", mmap_mode="r"),
            "soft": np.load(d / "soft2048.npy", mmap_mode="r"),
            "skel": np.load(d / "skel2048.npy", mmap_mode="r"),
            "disk": np.load(d / "disk.npy"),
            "bank": bank}
