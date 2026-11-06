"""Selection rules over analysis/metrics.jsonl: which photos are good. As a module: select(rows) -> rows with
"keep" and "reason". As a script: summary per object + review sheets (mask outline green = kept,
red = rejected, with the reason) for a random sample."""
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).parent
WORK = Path(__file__).resolve().parents[1] / "data" / "museum_oltenia" / "analysis"  # metrics, masks, series.json, sizes.json
MIN_EXTENT = 0.15   # object's longest side / photo's longest side
MIN_SHARP = 0.35    # Laplacian variance / the series' median
MAX_LEAK = 0.12     # the mask may reach this share of the detection's size beyond the detection box
MAX_ELEVATION = 0.80
MAX_FLARE = 1.6     # an upright object's mask may widen this much below its narrowest low row (a foot ring);
                    # wider, the mask has taken in the turntable it stands on  # turntable box height / width: above it the camera looks down on an upright object
BURST = re.compile(r"\(\d+\)$")  # 20251129_133430(0).jpg: a second copy of 20251129_133430.jpg
# objects that are white themselves (marble, white plaster restorations): a white mask is them, not the turntable
WHITE_OBJECTS = {"bust_marble_I7421", "strainer_I7769", "pot_I56504", "idol_dog_I7522"}
MAX_WIDE = 1.55  # mask width / the series' median width of kept masks
SAME_VIEW_IOU = 0.985  # consecutive frames whose object masks overlap this much show the same view


def load() -> list[dict]:
    return [json.loads(l) for l in open(WORK / "metrics.jsonl", encoding="utf-8")]


def select(rows: list[dict], shapes: dict[str, str]) -> list[dict]:
    by_series = defaultdict(list)
    for r in rows:
        by_series[(r["object"], r["setting"])].append(r)
    dark = {}
    if (WORK / "colours.jsonl").exists():
        dark = {d["path"]: d for d in map(json.loads, open(WORK / "colours.jsonl", encoding="utf-8"))}
    for (obj, _), group in by_series.items():
        laps = [r["lap"] for r in group if r.get("status") == "ok"]
        median = float(np.median(laps)) if laps else 0.0
        for r in group:
            r["keep"], r["reason"] = False, ""
            if r.get("status") != "ok":
                r["reason"] = r.get("status", "?")
                continue
            w, h = r["size"]
            x1, y1, x2, y2 = r["mask_box"]
            if r["border"]:
                r["reason"] = "cut by the edge"
            elif r["people"]:
                r["reason"] = "person"
            elif max(x2 - x1, y2 - y1) < MIN_EXTENT * max(w, h):
                r["reason"] = "too small"
            elif r["lap"] < MIN_SHARP * median:
                r["reason"] = "blurred"
            elif "table_box" in r and shapes[obj] == "upright" and _elevation(r["table_box"]) > MAX_ELEVATION:
                r["reason"] = "seen from above"
            elif _leak(r["mask_box"], r["det"][2:]) > MAX_LEAK:
                r["reason"] = "mask beyond the object"
            elif shapes[obj] == "upright" and _flare(_mask(r)) > MAX_FLARE:
                r["reason"] = "turntable in the mask"
            elif r.get("white_bottom", 0) > 0.5 and r["white_bottom"] - r.get("white_all", 0) > 0.3:
                r["reason"] = "white turntable in the mask"
            elif r.get("white_all", 0) > 0.6 and obj not in WHITE_OBJECTS:
                r["reason"] = "white turntable instead of the object"
            elif r["path"] in dark and dark[r["path"]]["dark_bottom"] > 0.55 and                     dark[r["path"]]["dark_bottom"] - dark[r["path"]]["dark_all"] > 0.35:
                r["reason"] = "black turntable in the mask"
            elif r.get("stand_cut", 0) > 0.4:
                r["reason"] = "mostly turntable"
            else:
                r["keep"] = True
        # a mask far wider than the series' usual one has taken in the turntable's whole top
        width = lambda r: (r["mask_box"][2] - r["mask_box"][0]) / max(r["size"])
        widths = [width(r) for r in group if r["keep"]]
        if widths:
            usual = float(np.median(widths))
            for r in group:
                if r["keep"] and width(r) > MAX_WIDE * usual:
                    r["keep"], r["reason"] = False, "turntable top in the mask"
        bursts = defaultdict(list)  # 20251129_133430.jpg and 20251129_133430(0).jpg: one shot, the sharpest kept
        for r in group:
            if r["keep"]:
                bursts[BURST.sub("", Path(r["path"]).stem)].append(r)
        for shots in bursts.values():
            for r in sorted(shots, key=lambda r: -r["lap"])[1:]:
                r["keep"], r["reason"] = False, "burst duplicate"
        last = None  # a frame of a continuous capture that shows the same view as the previous kept one
        for r in sorted((r for r in group if r["keep"]), key=lambda r: Path(r["path"]).stem):
            m = _mask(r) > 0
            if last is not None and m.shape == last.shape and (m & last).sum() / max(1, (m | last).sum()) >= SAME_VIEW_IOU:
                r["keep"], r["reason"] = False, "same view"
            else:
                last = m
    return rows


def _mask(r) -> np.ndarray:
    return cv2.imread(str(WORK / "masks" / f"{r['object']}__{r['setting']}__{Path(r['path']).stem}.png"),
                      cv2.IMREAD_GRAYSCALE)


def _flare(mask: np.ndarray) -> float:
    """Widest row below the narrowest row of the mask's lower 45% / that narrowest row."""
    rows = np.nonzero(mask.any(1))[0]
    if len(rows) < 20:
        return 0.0
    m = mask[rows[0]:rows[-1] + 1] > 0
    first, last = m.argmax(1), m.shape[1] - 1 - m[:, ::-1].argmax(1)
    widths = np.where(m.any(1), last - first + 1, 0)
    n = len(widths)
    low = widths[int(0.55 * n):int(0.97 * n)]  # the last rows are the rounded bottom edge
    if len(low) < 5:
        return 0.0
    k = int(np.argmin(low))
    return float(low[k:].max() / max(1, low[k]))


def _leak(mask_box, det_box) -> float:
    """How far the mask reaches beyond the detection box, as a share of the box's width / height."""
    w, h = max(1.0, det_box[2] - det_box[0]), max(1.0, det_box[3] - det_box[1])
    return max((det_box[0] - mask_box[0]) / w, (mask_box[2] - det_box[2]) / w,
               (det_box[1] - mask_box[1]) / h, (mask_box[3] - det_box[3]) / h)


def _elevation(box) -> float:
    return (box[3] - box[1]) / max(1, box[2] - box[0])


def _merged(obj, table) -> bool:
    area = lambda b: max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    return area(obj) > 0.8 * area(table)


def review(rows, out: Path, per_object: int = 8, seed: int = 0) -> None:
    random.seed(seed)
    by_obj = defaultdict(list)
    for r in rows:
        by_obj[r["object"]].append(r)
    sheets, tiles = 0, []
    for obj in sorted(by_obj):
        for r in random.sample(by_obj[obj], min(per_object, len(by_obj[obj]))):
            im = cv2.imdecode(np.fromfile(r["path"], np.uint8), cv2.IMREAD_REDUCED_COLOR_4)
            mp = WORK / "masks" / f"{r['object']}__{r['setting']}__{Path(r['path']).stem}.png"
            col = (0, 200, 0) if r["keep"] else (0, 0, 255)
            if mp.exists():
                m = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
                cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(im, cs, -1, col, 4)
            if "table_box" in r:
                cv2.rectangle(im, tuple(r["table_box"][:2]), tuple(r["table_box"][2:]), (255, 160, 0), 2)
            cv2.putText(im, (r["object"][:22] + " " + (r["reason"] or "kept")), (8, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.9, col, 2)
            h = 300
            tiles.append(cv2.resize(im, (int(im.shape[1] * h / im.shape[0]), h)))
            if len(tiles) == 48:
                _save(tiles, out / f"review_{sheets:02d}.jpg"); sheets += 1; tiles = []
    if tiles:
        _save(tiles, out / f"review_{sheets:02d}.jpg")


def _save(tiles, path):
    rows, row, w = [], [], 0
    for t in tiles:
        if w + t.shape[1] > 1900:
            rows.append(row); row, w = [], 0
        row.append(t); w += t.shape[1]
    rows.append(row)
    W = max(sum(t.shape[1] for t in r) for r in rows)
    sheet = np.vstack([cv2.copyMakeBorder(np.hstack(r), 2, 2, 0, W - sum(t.shape[1] for t in r),
                                          cv2.BORDER_CONSTANT, value=(255, 255, 255)) for r in rows])
    cv2.imwrite(str(path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 80])


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    shapes = {s["object"]: s["shape"] for s in json.load(open(WORK / "series.json", encoding="utf-8"))}
    rows = select(load(), shapes)
    reasons = defaultdict(int)
    for r in rows:
        reasons[r["reason"] or "kept"] += 1
    print(len(rows), "photos:", dict(sorted(reasons.items(), key=lambda kv: -kv[1])))
    per = defaultdict(lambda: [0, 0])
    for r in rows:
        per[r["object"]][0] += 1; per[r["object"]][1] += r["keep"]
    print(" ".join(f"{k}:{v[1]}/{v[0]}" for k, v in sorted(per.items())))
    if len(sys.argv) > 1:
        out = HERE / sys.argv[1]; out.mkdir(exist_ok=True)
        review(rows, out, int(sys.argv[2]) if len(sys.argv) > 2 else 8)
