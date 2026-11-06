"""The Met objects of the live demo seen from their other sides: The Met photographed most of its objects
from several angles (front, profile, back - the "additional images" of its Open Access records). For the 19
objects with more than one view of the whole object, in the same state (no details, no sword drawn out of
its scabbard), those photos are cut out like the main one (build_catalog.cut_out: YOLOv8m + SAM 2.1 +
matting) and the object turns in the hand through them, as the Muzeul Olteniei objects do: each real view
held DWELL steps, then TRANSITION steps of a cross-fade to the next (common/handheld.py VIEW_S per step).

Every view is scaled to the main cut-out's height (an object lying lengthwise: its length) and placed on
one canvas with it - upright objects bottom-centred (they stand on the palm), lying ones centred - so that
the object keeps its size and place while it turns. The views are at most VIEW_PX tall (the main view
stays at full resolution for the first view); their colours are bled into the transparent pixels (as
handheld._bleed), so the demo reads them as they are.

Saves photos/open_access/views/<object>_<k>.jpg (the Met's photo k of the object, at most as large as the
main photo), data/objects/views/<object>/<n>.png and data/objects/met_views.json ({object: [views]}),
which load_catalog() reads; --sheet also saves data/objects/met_views_preview.jpg (the photos of each object).

Usage: python met_views.py [--force] [--sheet]"""
import argparse
import json
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np

from build_catalog import cut_out
from common.handheld import CATALOG_DIR, _bleed, catalog_entries, sprite_path
from common.imaging import read_image, write_image
from common.paths import IMAGES_DIR
from common.segmentation import load_yolo

API = "https://collectionapi.metmuseum.org/public/collection/v1/objects/"
PHOTOS = IMAGES_DIR / "views"
VIEWS_DIR = CATALOG_DIR / "views"
OUT = CATALOG_DIR / "met_views.json"
VIEW_PX = 1200     # the tallest view, pixels (the main cut-out itself is up to 2400)
DWELL = 4          # steps each real view is held
TRANSITION = 4     # steps of the cross-fade between two views
# object -> (its Met record, the photos of the record that show the whole object, in turning order; the
# first is the catalog's photo)
MET = {
    "amphora_black_figure": (247236, [0, 1]),
    "amphora_red_figure": (247265, [0, 1]),
    "jar_terracotta": (240926, [0, 1]),
    "amphora_small_geometric": (251269, [0, 5, 4, 3, 2, 1]),
    "amphora_small_etruscan": (251468, [0, 1]),
    "amphora_large_attic": (254944, [0, 1, 2, 3]),
    "amphora_large_apulian": (247957, [0, 1]),
    "hippopotamus_william": (544227, [1, 0, 5, 2, 3]),
    "aquamanile_lion": (471288, [0, 1, 2]),
    "helmet_burgonet": (22634, [0, 4, 6, 5, 1]),
    "celestial_globe": (193606, [1, 0, 2, 3]),
    "helmet_corinthian": (703054, [1, 0, 3, 2]),
    "helmet_morion": (26450, [0, 9, 10, 2, 1]),
    "bust_roman": (248142, [0, 4, 1, 2, 3]),
    "pistol_flintlock": (33322, [0, 2, 1, 3]),
    "gauntlet_mitten": (22771, [2, 3, 4]),
    "war_hammer": (34089, [0, 1]),
    "armor_infantry": (22279, [0, 1]),
}
# left out after the review sheet: the presentation saber's other side curves the other way - the cross-fade
# between the two drew an X, not a turning blade
BOXES: dict[tuple[str, int], list[float]] = {}  # (object, photo): where it is, if YOLO does not find it


def fetch(name: str, oid: int, k: int, main: np.ndarray) -> Path:
    """The record's photo k, at most as large as the main photo (CREDITS.md rule)."""
    path = PHOTOS / f"{name}_{k}.jpg"
    if path.exists():
        return path
    req = lambda url: urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "dissertation-research"}),
                                             timeout=120).read()
    record = json.loads(req(API + str(oid)))
    url = ([record["primaryImage"]] + record.get("additionalImages", []))[k]
    image = cv2.imdecode(np.frombuffer(req(url), np.uint8), cv2.IMREAD_COLOR)
    scale = max(main.shape[:2]) / max(image.shape[:2])
    if scale < 1:
        image = cv2.resize(image, (round(image.shape[1] * scale), round(image.shape[0] * scale)), interpolation=cv2.INTER_AREA)
    write_image(path, image)
    time.sleep(0.2)
    return path


def premultiplied(sprite: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Float premultiplied BGRA at size (w, h)."""
    pm = sprite.astype(np.float32)
    pm[..., :3] *= pm[..., 3:] / 255
    return cv2.resize(pm, size, interpolation=cv2.INTER_AREA if size[1] < sprite.shape[0] else cv2.INTER_LINEAR)


def on_canvas(views: list[np.ndarray], lengthwise: bool) -> tuple[list[np.ndarray], tuple[int, int]]:
    """The views at one size (the first's height, or its width for an object lying lengthwise, at most
    VIEW_PX) on one canvas as wide and tall as the largest: premultiplied float BGRA."""
    first = views[0]
    if lengthwise:  # the same length
        length = min(first.shape[1], round(VIEW_PX * first.shape[1] / max(first.shape[:2])))
        sizes = [(length, max(1, round(v.shape[0] * length / v.shape[1]))) for v in views]
    else:  # the same height
        height = min(first.shape[0], VIEW_PX)
        sizes = [(max(1, round(v.shape[1] * height / v.shape[0])), height) for v in views]
    w, h = max(s[0] for s in sizes), max(s[1] for s in sizes)
    canvases = []
    for v, (vw, vh) in zip(views, sizes):
        c = np.zeros((h, w, 4), np.float32)
        x, y = (w - vw) // 2, (h - vh) // 2 if lengthwise else h - vh  # centred / standing on the bottom
        c[y:y + vh, x:x + vw] = premultiplied(v, (vw, vh))
        canvases.append(c)
    return canvases, (w, h)


def straight(pm: np.ndarray) -> np.ndarray:
    """Premultiplied float BGRA -> straight uint8 BGRA cropped to the object (one pixel of margin)."""
    alpha = pm[..., 3]
    out = pm.copy()
    out[..., :3] /= np.maximum(alpha[..., None] / 255, 1e-3)
    ys, xs = np.nonzero(alpha > 1)
    out = out[max(0, ys.min() - 1):ys.max() + 2, max(0, xs.min() - 1):xs.max() + 2]
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)


def between(a: np.ndarray, b: np.ndarray, t: float, lengthwise: bool) -> np.ndarray:
    """A step of the turn from view a to view b (straight uint8 BGRA, same height - or, lying, same length):
    both stretched to the width (lying: thickness) between theirs, so that their outlines meet, and
    cross-faded, t of the way."""
    if lengthwise:
        w, h = a.shape[1], max(1, round((1 - t) * a.shape[0] + t * b.shape[0]))
    else:
        w, h = max(1, round((1 - t) * a.shape[1] + t * b.shape[1])), a.shape[0]
    mixed = (1 - t) * premultiplied(a, (w, h)) + t * premultiplied(b, (w, h))
    return straight(mixed)


def main() -> None:
    parser = argparse.ArgumentParser(description="The Met objects' other views, for the live demo")
    parser.add_argument("--force", action="store_true", help="Cut the views out again")
    parser.add_argument("--sheet", action="store_true", help="Also save met_views_preview.jpg")
    args = parser.parse_args()
    entries = {sprite_path(e).stem: e for e in catalog_entries()}
    yolo, result, rows = None, {}, []
    for name, (oid, photos) in MET.items():
        entry = entries[name]
        main_photo = read_image(IMAGES_DIR / entry["photo"])
        folder = VIEWS_DIR / name
        if args.force or not (folder / "00.png").exists():
            yolo = yolo or load_yolo("m")
            views = [cv2.imread(str(sprite_path(entry)), cv2.IMREAD_UNCHANGED)]  # the main cut-out
            for k in photos[1:]:
                image = read_image(fetch(name, oid, k, main_photo))
                print(f"{name}: photo {k}")
                views.append(cut_out(yolo, image, entry.get("class", "vase"), BOXES.get((name, k), entry.get("box")),
                                     entry.get("rotate", 0)))
            lengthwise = "width_m" in entry
            canvases, _ = on_canvas(views, lengthwise)
            canvases = [straight(c) for c in canvases]
            steps = []
            for i, c in enumerate(canvases):
                steps += [c] * DWELL
                if i + 1 < len(canvases):  # to the next view: the outline goes from one to the other as both fade
                    steps += [between(c, canvases[i + 1], t, lengthwise)
                              for t in (np.arange(1, TRANSITION + 1) / (TRANSITION + 1))]
            if folder.exists():
                for old in folder.glob("*.png"):
                    old.unlink()
            folder.mkdir(parents=True, exist_ok=True)
            unique = []  # each image once: a view held DWELL steps is one file
            for s in steps:
                if not unique or s is not unique[-1][0]:
                    unique.append((s, len(unique)))
                    cv2.imwrite(str(folder / f"{len(unique) - 1:02d}.png"), _bleed(s), [cv2.IMWRITE_PNG_COMPRESSION, 9])
            order, last, n = [], None, -1
            for s in steps:
                if s is not last:
                    n += 1
                    last = s
                order.append(n)
            (folder / "order.json").write_text(json.dumps(order), encoding="utf-8")
        order = json.loads((folder / "order.json").read_text(encoding="utf-8"))
        result[name] = [name if k == 0 else f"views/{name}/{k:02d}" for k in order]  # view 0: the main cut-out
        if args.sheet:
            tiles = []
            for k in sorted(set(order), key=order.index):
                if order.count(k) < DWELL:  # a step between two photos
                    continue
                v = cv2.imread(str(folder / f"{k:02d}.png"), cv2.IMREAD_UNCHANGED)
                a = v[..., 3:] / 255.0
                t = (a * v[..., :3] + (1 - a) * 200).astype(np.uint8)
                tiles.append(cv2.resize(t, (max(1, round(t.shape[1] * 160 / t.shape[0])), 160), interpolation=cv2.INTER_AREA))
            row = np.hstack(tiles)
            rows.append(np.pad(row, ((0, 4), (0, 0), (0, 0)), constant_values=255))
    OUT.write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(f"Saved: {OUT} ({len(result)} objects)")
    if args.sheet:
        width = max(r.shape[1] for r in rows)
        rows = [np.pad(r, ((0, 0), (0, width - r.shape[1]), (0, 0)), constant_values=255) for r in rows]
        print("Saved:", write_image(CATALOG_DIR / "met_views_preview.jpg", np.vstack(rows)))


if __name__ == "__main__":
    main()
