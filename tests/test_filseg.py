"""Checks for the parts that are easy to get subtly wrong.

    python -m pytest tests -q
"""
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
from pycocotools import mask as mask_util

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from filseg import scorers, submission  # noqa: E402
from filseg.augment import dihedral, dihedral_inv  # noqa: E402
from filseg.candidates import extract  # noqa: E402
from filseg.metrics import iou_matrix, panoptic_quality, pq_from_table  # noqa: E402
from filseg.rle import NATIVE, decode, encode, from_polygons, to_rle  # noqa: E402

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


def blob(cx, cy, rx, ry):
    yy, xx = np.ogrid[:NATIVE, :NATIVE]
    return ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2 <= 1


def rle(mask):
    return to_rle(encode(mask))


# ---------------------------------------------------------------------------------- encoding
def test_rle_round_trip_is_exact():
    m = blob(1000, 700, 123, 17) | blob(10, 2040, 30, 30)
    assert np.array_equal(decode(encode(m)), m)


def test_polygon_is_not_transposed():
    m = decode(from_polygons([[100.0, 100.0, 400.0, 120.0, 380.0, 300.0, 120.0, 260.0]])["counts"])
    assert m[200, 250] and not m[250, 50]          # row is y, column is x


# ------------------------------------------------------------------------------------ metric
def dense_pq(gt: dict, pred: dict) -> float:
    """The metric written out with full arrays, as in the host's self-evaluation notebook."""
    tp_iou, fp, fn = [], 0, 0
    for rid, g in gt.items():
        p = pred.get(rid.split("-", 1)[1], [])
        if not p:
            fn += len(g)
            continue
        G = np.stack([mask_util.decode(x).ravel() for x in g]).astype(np.float64)
        P = np.stack([mask_util.decode(x).ravel() for x in p]).astype(np.float64)
        inter = G @ P.T
        iou = inter / (G.sum(1)[:, None] + P.sum(1)[None] - inter)
        hit = iou > 0.5
        tp_iou += iou[hit].tolist()
        fp += int((hit.sum(0) == 0).sum())
        fn += int((hit.sum(1) == 0).sum())
    return sum(tp_iou) / (len(tp_iou) + 0.5 * fp + 0.5 * fn)


@pytest.fixture(scope="module")
def toy():
    a, b, c = blob(300, 300, 60, 20), blob(900, 500, 30, 90), blob(1500, 1400, 80, 40)
    gt = {"010101-F": [rle(a), rle(b), rle(c)], "010102-F": [rle(a), rle(blob(905, 505, 28, 85))]}
    pred = [rle(blob(302, 301, 58, 21)), rle(blob(900, 520, 30, 60)), rle(blob(100, 1800, 20, 20))]
    return gt, pred


def test_pq_matches_the_dense_version(toy):
    gt, pred = toy
    assert panoptic_quality(gt, {"F": pred}).pq == pytest.approx(dense_pq(gt, {"F": pred}), abs=1e-9)


def test_no_prediction_scores_zero_and_a_copy_of_the_labels_scores_one(toy):
    gt, _ = toy
    assert panoptic_quality(gt, {}).pq == 0.0
    one = {"010101-F": gt["010101-F"]}
    assert panoptic_quality(one, {"F": one["010101-F"]}).pq == pytest.approx(1.0)


def test_pq_from_the_candidate_table_equals_the_metric(toy):
    gt, pred = toy
    mats = [iou_matrix(g, pred) for g in gt.values()]
    table = pd.DataFrame({
        "sum_iou": [sum(m[:, j][m[:, j] > 0.5].sum() for m in mats) for j in range(len(pred))],
        "tp": [sum(int((m[:, j] > 0.5).sum()) for m in mats) for j in range(len(pred))],
        "rec_hit": [sum(int((m[:, j] > 0.5).any()) for m in mats) for j in range(len(pred))],
        "n_rec": [len(gt)] * len(pred)})
    total = sum(len(v) for v in gt.values())
    for keep in ([1, 1, 1], [1, 0, 1], [0, 1, 0], [1, 1, 0]):
        k = np.array(keep, bool)
        exact = panoptic_quality(gt, {"F": [p for p, kk in zip(pred, k) if kk]}).pq
        assert pq_from_table(table, k, total) == pytest.approx(exact)


# -------------------------------------------------------------------------------- candidates
def test_candidates_are_disjoint_components_above_the_area_floor():
    prob = np.zeros((NATIVE, NATIVE), np.float32)
    prob[500:520, 400:700] = 0.95        # a filament
    prob[530:540, 400:700] = 0.9         # a second one, 10 px away
    prob[1500:1504, 1500:1504] = 0.99    # 16 px: below the floor
    img = np.full((NATIVE, NATIVE), 120, np.uint8)
    cands = extract(torch.from_numpy(prob), img, (1024.0, 1024.0, 950.0), threshold=0.8, min_area=30)
    assert len(cands) == 2
    masks = [mask_util.decode(c.mask_rle).astype(bool) for c in cands]
    assert not (masks[0] & masks[1]).any()
    assert {int(m.sum()) for m in masks} == {20 * 300, 10 * 300}
    assert all(c.feats["n_comp_img"] == 2 for c in cands)


def test_every_flip_is_undone_by_its_inverse():
    x = torch.arange(2 * 3 * 4 * 4, dtype=torch.float32).view(2, 3, 4, 4)
    for k in range(8):
        assert torch.equal(dihedral_inv(dihedral(x, k), k), x)


# ------------------------------------------------------------------------------ patch scorer
def test_patches_show_the_candidate_in_a_tight_and_a_wide_window():
    img = np.random.default_rng(0).integers(0, 255, (NATIVE, NATIVE), dtype=np.uint8)
    mask = np.zeros((NATIVE, NATIVE), bool)
    mask[1000:1008, 300:500] = True
    prob8 = np.where(mask, 230, 0).astype(np.uint8)
    cand = SimpleNamespace(mask_rle=rle(mask), bbox=(300, 1000, 200, 8))
    p = scorers.patches(img, prob8, [cand])
    assert p.shape == (1, 6, scorers.PATCH, scorers.PATCH) and p.dtype == np.uint8
    tight, wide = p[0, 2] > 127, p[0, 5] > 127
    assert tight.sum() > wide.sum() > 0


def test_a_window_outside_the_frame_is_zero():
    arr = np.full((NATIVE, NATIVE), 200, np.uint8)
    w = scorers._window(arr, 0, 0, 256)
    assert w[:60, :60].max() == 0 and w[-30:, -30:].min() == 200


@cuda
def test_patch_scorer_survives_a_save_and_load(tmp_path):
    net = scorers.patch_net("resnet18.a1_in1k", pretrained=False).cuda().eval()
    x = torch.randint(0, 255, (4, scorers.CHANNELS, scorers.PATCH, scorers.PATCH), dtype=torch.uint8)
    scorers.save_patch_scorer(net, tmp_path / "net.safetensors")
    again = scorers.load_patch_scorer("resnet18.a1_in1k", tmp_path / "net.safetensors")
    a, b = scorers.patch_scores([net], x), scorers.patch_scores([again], x)
    assert np.abs(a - b).max() < 0.02          # the weights are stored in half precision


# -------------------------------------------------------------------------------- submission
def test_submission_check_rejects_overlapping_masks():
    a, b = blob(500, 500, 40, 10), blob(520, 500, 40, 10)
    ok = pd.DataFrame(submission.rows_for_frame("F", [rle(a)]), columns=["filament_id", "segmentation_rle"])
    assert submission.check(ok, ["F"])["filaments"] == 1
    bad = pd.DataFrame(submission.rows_for_frame("F", [rle(a), rle(b)]), columns=["filament_id", "segmentation_rle"])
    with pytest.raises(AssertionError, match="overlapping"):
        submission.check(bad, ["F"])
