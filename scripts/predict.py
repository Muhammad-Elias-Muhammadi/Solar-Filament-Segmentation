"""Predict the test frames and write the submission.

    python scripts/predict.py                         # writes submission/submission.csv
    python scripts/predict.py --out my.csv --tta 8

Only the test image itself goes into the model. Reads from weights/: the network trained on all
frames, both keep-scorers and postprocess.json (threshold, tau, blend).
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from filseg import submission  # noqa: E402
from filseg.config import load_config  # noqa: E402
from filseg.data import read_image  # noqa: E402
from filseg.pipeline import Segmenter  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--tta", type=int, choices=[1, 2, 4, 8], default=None)
    args = ap.parse_args()
    cfg = load_config()
    segment = Segmenter(cfg)
    if args.tta:
        segment.tta = args.tta

    files = sorted(cfg["paths"]["test_images"].glob("*.jpeg"))
    rows, start = [], time.time()
    for f in tqdm(files, desc="test frames", unit="frame"):
        rows += submission.rows_for_frame(f.stem, [c.mask_rle for c in segment(read_image(f))])

    out = Path(args.out) if args.out else cfg["paths"]["submission"]
    df = submission.write(rows, out)
    report = {**submission.check(df, [f.stem for f in files]),
              "seconds_per_frame": round((time.time() - start) / len(files), 2),
              "md5": hashlib.md5(out.read_bytes()).hexdigest()}
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
