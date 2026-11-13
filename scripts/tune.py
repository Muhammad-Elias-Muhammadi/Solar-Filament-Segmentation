"""Choose the threshold and tau, and fit both keep-scorers, on the out-of-fold maps.

    python scripts/tune.py             # prints out-of-fold PQ for every threshold
    python scripts/tune.py --final     # also writes the scorers and postprocess.json to weights/

Needs the out-of-fold maps written by scripts/train.py. Every number printed is pooled PQ with the
host's metric on all 707 training frames: each frame's candidates come from the network that never
saw the frame, and both scorers are fitted on four folds and applied to the fifth. Nothing here
looks at the test set or the leaderboard.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from filseg import scorers  # noqa: E402
from filseg.candidates import extract  # noqa: E402
from filseg.config import load_config  # noqa: E402
from filseg.data import load_cache, load_records  # noqa: E402
from filseg.folds import load_folds  # noqa: E402
from filseg.metrics import GroundTruth, candidate_table, pq_from_table  # noqa: E402
from filseg.inference import on_disk  # noqa: E402

TAUS = np.round(np.arange(0.10, 0.36, 0.01), 2)


def out_of_fold_maps(runs: Path) -> dict[str, tuple[np.ndarray, int]]:
    maps = {}
    for f in sorted(runs.glob("oof_fold*_stems.json")):
        arr = np.load(str(f).replace("_stems.json", ".npy"), mmap_mode="r")
        maps.update({s: (arr, i) for i, s in enumerate(json.loads(f.read_text()))})
    if not maps:
        raise FileNotFoundError(f"no out-of-fold maps in {runs}; run scripts/train.py first")
    return maps


def candidates_of_all_frames(cfg: dict, thresholds: list[float]) -> tuple[pd.DataFrame, np.ndarray, int]:
    """One table row and one patch per candidate, for every threshold."""
    cache = load_cache(cfg)
    maps = out_of_fold_maps(cfg["paths"]["runs"])
    gt = GroundTruth(load_records(cfg))
    index = {s: i for i, s in enumerate(cache["stems"])}
    stems = [s for s in cache["stems"] if s in maps]
    tables, pats = [], []
    for stem in tqdm(stems, desc="candidates", unit="frame"):
        arr, j = maps[stem]
        img, disk = np.asarray(cache["img"][index[stem]]), tuple(cache["disk"][index[stem]])
        prob = on_disk(torch.as_tensor(np.asarray(arr[j], np.float32) / 255.0, device="cuda"), disk)
        prob8 = (prob.cpu().numpy() * 255).round().astype(np.uint8)
        for thr in thresholds:
            cands = extract(prob, img, disk, thr)
            if cands:
                tables.append(candidate_table(stem, cands, gt))
                pats.append(scorers.patches(img, prob8, cands))
    return pd.concat(tables, ignore_index=True), np.concatenate(pats), gt.n_filaments(stems)


def best_tau(table: pd.DataFrame, score: np.ndarray, n_filaments: int) -> tuple[float, float]:
    return max((pq_from_table(table, score > tau, n_filaments), float(tau)) for tau in TAUS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--final", action="store_true", help="write the scorers and postprocess.json to weights/")
    args = ap.parse_args()
    cfg = load_config()
    tcfg, pcfg = cfg["tune"], cfg["patch_scorer"]
    out = cfg["paths"]["tuning"]
    out.mkdir(parents=True, exist_ok=True)

    cached = out / "candidates.pkl"
    if cached.exists():
        table, pats = pd.read_pickle(cached), np.load(out / "patches.npy")
        n_filaments = json.loads((out / "meta.json").read_text())["n_filaments"]
    else:
        table, pats, n_filaments = candidates_of_all_frames(cfg, tcfg["thresholds"])
        table.to_pickle(cached)
        np.save(out / "patches.npy", pats)
        (out / "meta.json").write_text(json.dumps({"n_filaments": int(n_filaments), "thresholds": tcfg["thresholds"]}))
    folds = table["stem"].map(load_folds(cfg)).to_numpy()
    print(f"{table['stem'].nunique()} frames, {len(table)} candidates, {n_filaments} annotated filaments", flush=True)

    # patch scorer, out of fold: trained on the candidates of all thresholds of four folds
    X = torch.as_tensor(pats, device="cuda")
    y = torch.as_tensor(table["y"].to_numpy(np.float32), device="cuda")
    seeds = list(range(pcfg["seeds"]))
    cnn = np.zeros(len(table), np.float32)
    for k in np.unique(folds):
        tr, va = np.flatnonzero(folds != k), np.flatnonzero(folds == k)
        cnn[va] = scorers.patch_scores([scorers.fit_patch_scorer(X, y, tr, pcfg, s) for s in seeds], X, va)
        print(f"  patch scorer, fold {k} done", flush=True)

    w, rows = pcfg["blend"], []
    for thr in tcfg["thresholds"]:
        sel = np.isclose(table["thr"].to_numpy(), thr)
        sub = table[sel].reset_index(drop=True)
        gbm = scorers.feature_scores_out_of_fold(sub, folds[sel], cfg["seed"])
        res = {"threshold": thr, "components_only": pq_from_table(sub, np.ones(len(sub), bool), n_filaments)}
        for key, score in (("features", gbm), ("patches", cnn[sel]), ("both", w * gbm + (1 - w) * cnn[sel])):
            res[key], res[f"tau_{key}"] = best_tau(sub, score, n_filaments)
        res["perfect_keep"] = max(pq_from_table(sub, sub["y"].to_numpy() > t, n_filaments) for t in (0.2, 0.25, 0.3))
        rows.append(res)
        print({k: round(v, 4) for k, v in res.items()}, flush=True)
    grid = pd.DataFrame(rows)
    grid.round(4).to_csv(out / "tuning.csv", index=False)
    best = grid.loc[grid["both"].idxmax()]
    print(f"best: threshold {best['threshold']}, tau {best['tau_both']}, out-of-fold PQ {best['both']:.4f}")
    if not args.final:
        return

    weights = cfg["paths"]["weights"]
    sub = table[np.isclose(table["thr"], best["threshold"])]
    scorers.save_pickle(scorers.feature_scorer(cfg["seed"]).fit(sub[scorers.FEATURES].to_numpy(np.float32), sub["y"].to_numpy()),
                        weights / "feature_scorer.pkl")
    files = []
    for s in seeds:                     # the final patch scorers see every out-of-fold candidate
        files.append(f"patch_scorer_seed{s}.safetensors")
        scorers.save_patch_scorer(scorers.fit_patch_scorer(X, y, np.arange(len(table)), pcfg, s), weights / files[-1])
    post = {"network": "unet_full.safetensors", "threshold": float(best["threshold"]), "min_area": 30,
            "keep_tau": float(best["tau_both"]), "blend": w, "feature_scorer": "feature_scorer.pkl",
            "patch_scorer": {"arch": pcfg["arch"], "files": files}, "out_of_fold_pq": round(float(best["both"]), 4)}
    (weights / "postprocess.json").write_text(json.dumps(post, indent=1))
    print(json.dumps(post, indent=1))


if __name__ == "__main__":
    main()
