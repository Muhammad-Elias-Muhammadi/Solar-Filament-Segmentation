"""Panoptic Quality as the competition computes it.

* The unit of evaluation is one annotator record of one frame. A predicted mask is looked up by
  frame, so it is scored once against every record of its frame.
* A predicted and an annotated filament match when their IoU is above 0.5. With disjoint
  predictions a filament can match at most one of them, so no assignment step is needed.
* Counts are pooled over all records: PQ = sum of matched IoU / (TP + FP/2 + FN/2).

IoUs are computed on run-length codes with pycocotools, which is exact and far cheaper than
comparing 2048 x 2048 arrays. tests/test_metrics.py checks it against a dense implementation.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from pycocotools import mask as mask_util

from .data import Record
from .rle import from_polygons

IOU_THR = 0.5


def iou_matrix(gt: list, pred: list) -> np.ndarray:
    """(n_gt, n_pred) IoU of two lists of RLE objects."""
    if not gt or not pred:
        return np.zeros((len(gt), len(pred)))
    return np.asarray(mask_util.iou(pred, gt, [0] * len(gt))).reshape(len(pred), len(gt)).T


class GroundTruth:
    """The annotated filaments as RLE objects, grouped by record and by frame."""

    def __init__(self, records: dict[str, list[Record]]):
        self.by_record = {r.record_id: [from_polygons(p) for p in r.polygons] for recs in records.values() for r in recs}
        self.by_frame = {stem: [r.record_id for r in recs] for stem, recs in records.items()}

    def records(self, stem: str) -> list[list]:
        return [self.by_record[r] for r in self.by_frame.get(stem, [])]

    def subset(self, stems) -> dict[str, list]:
        return {r: self.by_record[r] for s in stems for r in self.by_frame.get(s, [])}

    def n_filaments(self, stems) -> int:
        return sum(len(v) for v in self.subset(stems).values())


@dataclass
class PQResult:
    pq: float
    sq: float
    rq: float
    tp: int
    fp: int
    fn: int
    per_record: pd.DataFrame

    def summary(self) -> dict:
        return {"PQ": round(self.pq, 4), "SQ": round(self.sq, 4), "RQ": round(self.rq, 4),
                "TP": self.tp, "FP": self.fp, "FN": self.fn}


def panoptic_quality(gt: dict[str, list], pred: dict[str, list]) -> PQResult:
    """`gt`: record id -> RLE objects; `pred`: frame name -> RLE objects."""
    matched, fp, fn, rows = [], 0, 0, []
    for rid, g in gt.items():
        stem = rid.split("-", 1)[1]
        p = pred.get(stem, [])
        m = iou_matrix(g, p)
        hit = m > IOU_THR
        ious = m[hit]
        fp_r = int((hit.sum(0) == 0).sum()) if g else len(p)
        fn_r = int((hit.sum(1) == 0).sum()) if p else len(g)
        matched += ious.tolist()
        fp, fn = fp + fp_r, fn + fn_r
        rows.append((rid, stem, len(ious), fp_r, fn_r, float(ious.sum())))
    tp = len(matched)
    denom = tp + 0.5 * fp + 0.5 * fn
    return PQResult(pq=float(sum(matched) / denom) if denom else 0.0, sq=float(np.mean(matched)) if tp else 0.0,
                    rq=float(tp / denom) if denom else 0.0, tp=tp, fp=fp, fn=fn,
                    per_record=pd.DataFrame(rows, columns=["record", "frame", "tp", "fp", "fn", "sum_iou"]))


def candidate_table(stem: str, cands: list, gt: GroundTruth) -> pd.DataFrame:
    """One row per candidate: its features and its exact contribution to pooled PQ.

    sum_iou is the sum of its matched IoUs over the records of the frame, tp the number of records
    it matches, n_rec the number of records, and y = sum_iou / n_rec the target of both scorers.
    """
    if not cands:
        return pd.DataFrame()
    recs = gt.records(stem)
    mats = [iou_matrix(g, [c.mask_rle for c in cands]) for g in recs]
    rows = []
    for j, c in enumerate(cands):
        sum_iou = tp = rec_hit = 0
        for m in mats:
            col = m[:, j] if m.size else np.zeros(0)
            hit = col > IOU_THR
            sum_iou += col[hit].sum()
            tp += int(hit.sum())
            rec_hit += int(hit.any())
        rows.append({**c.feats, "stem": stem, "sum_iou": sum_iou, "tp": tp, "rec_hit": rec_hit,
                     "n_rec": len(recs), "y": sum_iou / max(1, len(recs))})
    return pd.DataFrame(rows)


def pq_from_table(table: pd.DataFrame, keep: np.ndarray, n_filaments: int) -> float:
    """Pooled PQ of any subset of disjoint candidates, without computing a single IoU again.

    With disjoint predictions every annotated filament is matched by at most one of them, so the
    pooled counts add up over the kept candidates.
    """
    tp = table["tp"].to_numpy()[keep].sum()
    fp = (table["n_rec"].to_numpy()[keep] - table["rec_hit"].to_numpy()[keep]).sum()
    denom = tp + 0.5 * fp + 0.5 * (n_filaments - tp)
    return float(table["sum_iou"].to_numpy()[keep].sum() / denom) if denom > 0 else 0.0
