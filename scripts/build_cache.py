"""Decode the training frames and rasterise their targets once (work/cache, about 9 GB).

    python scripts/build_cache.py
"""
import argparse
import sys
import time
from pathlib import Path

from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from filseg.config import load_config  # noqa: E402
from filseg.data import build_cache, load_cache  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()
    cfg = load_config()
    start = time.time()
    out = build_cache(cfg, args.workers, progress=tqdm)
    cache = load_cache(cfg)
    print(f"{out}: {len(cache['stems'])} frames, {len(cache['bank'])} filaments for copy-paste, {time.time() - start:.0f} s")


if __name__ == "__main__":
    main()
