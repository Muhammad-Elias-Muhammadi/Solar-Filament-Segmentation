"""Train the network on the five folds and on all frames.

    python scripts/train.py                      # folds 0 to 4, then all frames (about 12 GPU hours)
    python scripts/train.py --folds 0            # one fold

A fold model is saved at its best validation PQ and then writes the maps of its validation frames
(work/runs/oof_fold<k>.npy), which scripts/tune.py needs. The model trained on all frames has no
validation frames and is saved after the last epoch; it is the one used on the test set.
"""
import argparse
import json
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from filseg.config import load_config  # noqa: E402
from filseg.data import load_cache  # noqa: E402
from filseg.training import train, write_out_of_fold  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", nargs="+", default=["0", "1", "2", "3", "4", "full"])
    ap.add_argument("--epochs", type=int, default=None, help="override the number of epochs (for a quick test)")
    args = ap.parse_args()
    cfg = load_config()
    cache = load_cache(cfg)
    runs, weights = cfg["paths"]["runs"], cfg["paths"]["weights"]
    for name in args.folds:
        fold = None if name == "full" else int(name)
        start = time.time()
        best = train(cfg, cache, fold, epochs=args.epochs)
        if fold is not None:
            write_out_of_fold(cfg, cache, fold)
        shutil.copy2(best["weights"], weights / f"unet_{'full' if fold is None else f'fold{fold}'}.safetensors")
        summary = runs / "summary.json"
        done = json.loads(summary.read_text()) if summary.exists() else {}
        done[name] = {**best, "minutes": round((time.time() - start) / 60, 1)}
        summary.write_text(json.dumps(done, indent=1))
        print(f"[{name}] best validation PQ {best['val_pq']:.4f} at epoch {best['epoch']}, "
              f"{(time.time() - start) / 60:.0f} min", flush=True)


if __name__ == "__main__":
    main()
