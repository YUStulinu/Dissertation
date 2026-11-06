"""Figure of the selection: for each outcome (kept, and the main reasons a photo was left out) a row of
example photos from different objects, the object's mask outlined (green = kept, red = left out).
Needs the original photos (the advisor's database), so it was made before that folder was deleted.

Usage: python collection_selection/selection_figure.py   -> data/museum_oltenia/figures/selection_reasons.jpg"""
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from selection import WORK, _mask, load, select  # noqa: E402

ROWS = [("kept", "kept"), ("cut by the edge", "cut by the edge"), ("blurred", "blurred"),
        ("person", "person / hand / foot in the photo"), ("too small", "object too small"),
        ("black turntable in the mask", "black turntable in the mask"),
        ("white turntable instead of the object", "white turntable instead of the object"),
        ("turntable top in the mask", "turntable top in the mask")]
PER_ROW = 6
TILE_H = 260


def tile(r) -> np.ndarray:
    im = cv2.imdecode(np.fromfile(r["path"], np.uint8), cv2.IMREAD_REDUCED_COLOR_4)
    m = _mask(r)
    if m is not None:
        cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(im, cs, -1, (60, 200, 60) if r["keep"] else (60, 60, 230), 5)
    s = TILE_H / im.shape[0]
    return cv2.resize(im, (round(im.shape[1] * s), TILE_H), interpolation=cv2.INTER_AREA)


def main() -> None:
    shapes = {s["object"]: s["shape"] for s in json.load(open(WORK / "series.json", encoding="utf-8"))}
    rows = select(load(), shapes)
    by = defaultdict(list)
    for r in rows:
        by[r["reason"] or "kept"].append(r)
    random.seed(3)
    width = PER_ROW * 350
    out = []
    for reason, title in ROWS:
        pool = by[reason]
        n = sum(1 for _ in pool)
        picks, seen = [], set()
        for r in random.sample(pool, len(pool)):  # one photo per object
            if r["object"] not in seen:
                picks.append(r); seen.add(r["object"])
            if len(picks) == PER_ROW:
                break
        tiles = [tile(r) for r in picks]
        strip = np.hstack([cv2.copyMakeBorder(t, 0, 0, 3, 3, cv2.BORDER_CONSTANT, value=(255, 255, 255)) for t in tiles])
        strip = cv2.resize(strip, (width, round(strip.shape[0] * width / strip.shape[1])), interpolation=cv2.INTER_AREA)
        bar = np.full((44, width, 3), 255, np.uint8)
        cv2.putText(bar, f"{title}: {n} photos", (8, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.95,
                    (40, 140, 40) if reason == "kept" else (40, 40, 200), 2, cv2.LINE_AA)
        out += [bar, strip, np.full((10, width, 3), 255, np.uint8)]
    path = WORK.parent / "figures" / "selection_reasons.jpg"
    path.parent.mkdir(exist_ok=True)
    cv2.imwrite(str(path), np.vstack(out), [cv2.IMWRITE_JPEG_QUALITY, 85])
    print("Saved:", path)


if __name__ == "__main__":
    main()
