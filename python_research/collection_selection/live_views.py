"""The Muzeul Olteniei objects in the live demo (ar_objects.py and the C++ app): for each object (81 of the 83:
LEFT_OUT have no photo of their own, only of the turntable), VIEWS cut-outs of it turned on the turntable, which the demo shows in turn - the object slowly turns back
and forth in the hand (common/handheld.py VIEW_S).

The views come from one turn of the turntable: the longest run of consecutive photos taken from the same
height (the turntable's box keeps its shape: its height / width changes by less than STEADY from one photo
to the next) and from the side (the object stands on the turntable's top edge, as in sizes.py), when there
are enough of those (otherwise, without a long turn, the largest group taken from one height); VIEWS of them, evenly spaced. Each is cut out of its photo with its mask (matted, as
harmonize_collection.py does), at most VIEW_PX tall, its colours bled into its transparent pixels (as
handheld._bleed) so the demo reads it as it is: data/objects/views/<object>/<k>.png, the first one also
data/objects/<object>.png (the object's main cut-out: its shape decides how it is held).

The size it is shown at is objects.csv's estimate; an object without one gets the median of its kind (the
first word of its name: cup, jug, axe...) or of all objects measured the same way, marked "approx_size" (the
demo writes "~"). A size given as a length (flat objects, measured from above) is the object's longest side:
its height is that length times the cut-outs' least height / width.

Writes data/objects/oltenia.json, which load_catalog() adds to catalog.json.

Usage: python collection_selection/live_views.py [--force]   (from python_research/)"""
import argparse
import csv
import json
import os
import shutil
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from common.handheld import CATALOG_DIR, _bleed  # noqa: E402
from common.imaging import read_image, read_mask, write_image  # noqa: E402
from common.paths import DATA_DIR, IMAGES_DIR  # noqa: E402

COLLECTION = DATA_DIR / "museum_oltenia"
PHOTOS = IMAGES_DIR.parent  # photos.csv paths start with museum_oltenia/
VIEWS_DIR = CATALOG_DIR / "views"
OUT = CATALOG_DIR / "oltenia.json"
VIEWS = 12        # views per object
VIEW_PX = 720     # the tallest view, pixels: a 25 cm object held at arm's length is ~500-700 px tall in Full HD
STEADY = 0.06     # one turn: the turntable box's height / width changes by less than this between two photos
SIDE_DROP = 0.3   # a side view: the object's bottom at most this share of the turntable's width below its top
MAX_GAP = 3       # one turn: photo numbers at most this far apart (the selection removed some)
# found on the review sheets of the views: their masks hold the turntable (the black disc on its stand, the
# white cylinder) or a lamp instead of the object - photos left out of the views, and objects without any
# good view (the silver hair ring has no photo at all)
NOT_VIEWS = {"spearhead_I56410_001_lab", "spearhead_I56410_030_white", "spearhead_iron_I42460_024_white",
             "spearhead_iron_I42460_035_white", "mould_axe_bivalve_I54478_073_white", "sword_iron_I7713_148_white",
             "sword_iron_I7713_161_white", "idol_bull_I26022_065_white", "clay_box_M38_2_073_white",
             "pot_18298_082_lab", "mould_axe_bivalve_I54478_074_white", "clay_box_M38_2_071_white"}
LEFT_OUT = {"fibula_bronze_55790", "ring_bronze_55814"}


def number(photo: str) -> int:
    return int(Path(photo).stem.rsplit("_", 2)[1])


def side(m: dict) -> bool:
    if "stand_box" not in m:
        return False
    sx1, sy1, sx2, _ = m["stand_box"]
    return (m["mask_box"][3] - sy1) / max(1, sx2 - sx1) < SIDE_DROP


def elevation(m: dict) -> float | None:
    if "stand_box" not in m:
        return None
    sx1, sy1, sx2, sy2 = m["stand_box"]
    return (sy2 - sy1) / max(1, sx2 - sx1)


def one_turn(items: list[tuple[dict, dict]]) -> list[tuple[dict, dict]]:
    """The longest run of consecutive photos (same backdrop, numbers close, the camera at the run's height:
    a photo taken from elsewhere in between is passed over)."""
    best, run = [], []
    for item in sorted(items, key=lambda it: (it[0]["setting"], number(it[0]["photo"]))):
        row, m = item
        if run:
            last = run[-1][0]
            if row["setting"] != last["setting"] or number(row["photo"]) - number(last["photo"]) > MAX_GAP:
                run = []
            elif abs(elevation(m) - float(np.median([elevation(x) for _, x in run]))) > STEADY:
                continue  # taken from another height: passed over
        run.append(item)
        if len(run) > len(best):
            best = list(run)
    return best


def same_height(items: list[tuple[dict, dict]]) -> list[tuple[dict, dict]]:
    """No long turn: the largest group of photos with one backdrop taken from about one height, in order."""
    groups = defaultdict(list)
    for item in items:
        groups[(item[0]["setting"], round(elevation(item[1]) / (2 * STEADY)))].append(item)
    group = max(groups.values(), key=len)
    return sorted(group, key=lambda it: number(it[0]["photo"]))


def pick(items: list[tuple[dict, dict]]) -> list[dict]:
    measured = [it for it in items if elevation(it[1]) is not None]
    sides = [it for it in measured if side(it[1])]
    pool = sides if len(sides) >= 4 else measured
    turn = one_turn(pool) if pool else []
    if len(turn) < min(VIEWS, len(items)):  # no full turn of views: the photos taken from one height
        turn = same_height(pool) if len(pool) >= min(VIEWS, len(items)) else sorted(
            items, key=lambda it: (it[0]["setting"], number(it[0]["photo"])))
    k = np.unique(np.linspace(0, len(turn) - 1, min(VIEWS, len(turn))).round().astype(int))
    return [turn[i][0] for i in k]


def make_view(job: tuple[str, str]) -> tuple[int, int]:
    """Cut out one view (in a worker process): -> its width, height."""
    from harmonize_collection import cut_out
    import cv2

    photo, out = job
    image = read_image(PHOTOS / photo)
    mask = read_mask(COLLECTION / "masks" / f"{Path(photo).stem}.png", image.shape[:2])
    sprite = cut_out(image, mask)
    if sprite.shape[0] > VIEW_PX:
        size = (max(1, round(sprite.shape[1] * VIEW_PX / sprite.shape[0])), VIEW_PX)
        pm = sprite.astype(np.float32)  # premultiplied, so no background colour comes into the edge
        pm[..., :3] *= pm[..., 3:] / 255
        pm = cv2.resize(pm, size, interpolation=cv2.INTER_AREA)
        pm[..., :3] /= np.maximum(pm[..., 3:] / 255, 1e-3)
        sprite = np.clip(np.rint(pm), 0, 255).astype(np.uint8)
    write_image(Path(out), _bleed(sprite))
    return sprite.shape[1], sprite.shape[0]


def main() -> None:
    parser = argparse.ArgumentParser(description="Turning views of the Muzeul Olteniei objects for the live demo")
    parser.add_argument("--force", action="store_true", help="Cut the views out again")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    metrics = {}
    for line in open(COLLECTION / "analysis" / "metrics.jsonl", encoding="utf-8"):
        m = json.loads(line)
        metrics[m["path"].split("Poze pentru disertatie\\", 1)[1].replace("\\", "/")] = m
    photos = defaultdict(list)
    for row in csv.DictReader(open(COLLECTION / "photos.csv", encoding="utf-8")):
        if row["object"] not in LEFT_OUT and Path(row["photo"]).stem not in NOT_VIEWS:
            photos[row["object"]].append((row, metrics[row["source"]]))
    objects = {r["object"]: r for r in csv.DictReader(open(COLLECTION / "objects.csv", encoding="utf-8"))}

    chosen = {obj: pick(items) for obj, items in sorted(photos.items())}
    before = {e["name"]: e.get("view_photos") for e in json.loads(OUT.read_text(encoding="utf-8"))} if OUT.exists() else {}
    for obj, photos_before in before.items():  # views chosen otherwise now (or the object left out): cut out again
        if obj not in chosen or photos_before != [Path(row["photo"]).name for row in chosen[obj]]:
            shutil.rmtree(VIEWS_DIR / obj, ignore_errors=True)
            if obj not in chosen:
                (CATALOG_DIR / f"{obj}.png").unlink(missing_ok=True)
    jobs = []
    for obj, rows in chosen.items():
        for k, row in enumerate(rows):
            out = VIEWS_DIR / obj / f"{k:02d}.png"
            if args.force or not out.exists():
                jobs.append((row["photo"], str(out)))
    print(f"{len(chosen)} objects, {sum(map(len, chosen.values()))} views, {len(jobs)} to cut out", flush=True)
    with ProcessPoolExecutor(max(1, (os.cpu_count() or 2) // 2)) as pool:
        for i, _ in enumerate(pool.map(make_view, jobs, chunksize=4)):
            if (i + 1) % 100 == 0:
                print(f"  {i + 1}/{len(jobs)}", flush=True)

    import cv2
    known = defaultdict(list)  # (kind of size, first word of the name) -> sizes measured
    for obj, r in objects.items():
        if r["size_m"]:
            known[(r["size_kind"], obj.split("_")[0])].append(float(r["size_m"]))
            known[(r["size_kind"], None)].append(float(r["size_m"]))
    entries = []
    for obj, rows in chosen.items():
        r = objects[obj]
        size = float(r["size_m"]) if r["size_m"] else float(np.median(
            known.get((r["size_kind"], obj.split("_")[0])) or known[(r["size_kind"], None)]))
        shapes = [cv2.imread(str(VIEWS_DIR / obj / f"{k:02d}.png"), cv2.IMREAD_UNCHANGED).shape[:2] for k in range(len(rows))]
        height = size if r["size_kind"] == "height" else size * min(1.0, min(h / w for h, w in shapes))
        main = cv2.imread(str(VIEWS_DIR / obj / "00.png"), cv2.IMREAD_UNCHANGED)
        cv2.imwrite(str(CATALOG_DIR / f"{obj}.png"), main)
        entry = {"name": obj, "photo": rows[0]["photo"], "height_m": round(height, 4), "size_m": round(size, 4),
                 "views": [f"views/{obj}/{k:02d}" for k in range(len(rows))], "hands": 1,
                 "view_photos": [Path(row["photo"]).name for row in rows]}
        if not r["size_m"]:
            entry["approx_size"] = True
        entries.append(entry)
    OUT.write_text("[\n" + ",\n".join("  " + json.dumps(e, ensure_ascii=False) for e in entries) + "\n]\n", encoding="utf-8")
    print(f"Saved: {OUT} ({len(entries)} objects, {sum(len(e['views']) for e in entries)} views, "
          f"{sum('approx_size' in e for e in entries)} sizes approximate)")


if __name__ == "__main__":
    main()
