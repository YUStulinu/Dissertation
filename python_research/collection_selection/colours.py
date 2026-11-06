"""Dark share of the whole mask and of its lowest 15% of rows, for the photos the rules keep so far:
the lab's black turntable taken into a mask makes its bottom far darker than the object.
Writes analysis/colours.jsonl."""
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).parent
WORK = Path(__file__).resolve().parents[1] / "data" / "museum_oltenia" / "analysis"  # metrics, masks, series.json, sizes.json
sys.path.insert(0, str(HERE))
from selection import load, select  # noqa: E402


def measure(r) -> dict:
    im = cv2.imdecode(np.fromfile(r["path"], np.uint8), cv2.IMREAD_REDUCED_COLOR_4)
    m = cv2.imread(str(WORK / "masks" / f"{r['object']}__{r['setting']}__{Path(r['path']).stem}.png"), 0) > 0
    v = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)[..., 2]
    dark = v < 60
    ys = np.nonzero(m.any(1))[0]
    low = m.copy()
    low[:ys[0] + int(0.85 * (ys[-1] - ys[0] + 1))] = False
    return {"path": r["path"], "dark_all": round(float(dark[m].mean()), 4),
            "dark_bottom": round(float(dark[low].mean()) if low.any() else 0.0, 4)}


if __name__ == "__main__":
    shapes = {s["object"]: s["shape"] for s in json.load(open(WORK / "series.json", encoding="utf-8"))}
    kept = [r for r in select(load(), shapes) if r["keep"] and r["setting"] != "white"]
    with ThreadPoolExecutor(8) as pool, open(WORK / "colours.jsonl", "w", encoding="utf-8") as out:
        for d in pool.map(measure, kept):
            out.write(json.dumps(d) + "\n")
    print("measured", len(kept))
