"""Copies the selected photos into the project: photos/museum_oltenia/<object>/<object>_<setting>_<NNN>.jpg
(longest side 1600 px, the rule of photos/CREDITS.md), the object's mask at the same size in
python_research/data/museum_oltenia/masks/<name>.png, and a manifest (CSV) of where each came from.
Every written photo is read back and checked. The originals are not touched here."""
import csv
import json
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).parent
WORK = Path(__file__).resolve().parents[1] / "data" / "museum_oltenia" / "analysis"  # metrics, masks, series.json, sizes.json
sys.path.insert(0, str(HERE))
from selection import load, select  # noqa: E402

PROJECT = Path(r"C:\MASTER\disertatie")
SOURCE = Path(r"C:\MASTER\Poze pentru disertatie")
PHOTOS = PROJECT / "photos" / "museum_oltenia"
DATA = PROJECT / "python_research" / "data" / "museum_oltenia"
LONG_SIDE = 1600


def setting_of(image: np.ndarray, mask: np.ndarray, given: str) -> str:
    """white (the white backdrop) or lab; for the series that mix both, from the photo's surroundings."""
    if given != "mixed":
        return given
    hsv = cv2.cvtColor(cv2.resize(image, (200, round(200 * image.shape[0] / image.shape[1]))), cv2.COLOR_BGR2HSV)
    around = cv2.resize(mask, hsv.shape[1::-1], interpolation=cv2.INTER_NEAREST) == 0
    return "white" if np.median(hsv[..., 2][around]) > 165 and np.median(hsv[..., 1][around]) < 40 else "lab"


def copy_one(job) -> dict:
    row, name_stem = job
    image = cv2.imdecode(np.fromfile(row["path"], np.uint8), cv2.IMREAD_COLOR)
    small_mask = cv2.imread(str(WORK / "masks" / f"{row['object']}__{row['setting']}__{Path(row['path']).stem}.png"),
                            cv2.IMREAD_GRAYSCALE)
    h, w = image.shape[:2]
    if abs(small_mask.shape[1] / small_mask.shape[0] - w / h) > 0.01:
        raise RuntimeError(f"mask and photo differ in shape: {row['path']}")
    scale = LONG_SIDE / max(h, w)
    size = (round(w * scale), round(h * scale))
    photo = cv2.resize(image, size, interpolation=cv2.INTER_AREA) if scale < 1 else image
    mask = np.where(cv2.resize(small_mask, photo.shape[1::-1], interpolation=cv2.INTER_LINEAR) > 127, 255, 0).astype(np.uint8)
    setting = setting_of(photo, mask, row["setting"])
    name = f"{name_stem}_{setting}"
    return {"photo": photo, "mask": mask, "name": name, "row": row, "original_size": f"{w}x{h}"}


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    series = json.load(open(WORK / "series.json", encoding="utf-8"))
    shapes = {s["object"]: s["shape"] for s in series}
    labels = {s["object"]: s["label"] for s in series}
    kept = [r for r in select(load(), shapes) if r["keep"]]
    by_obj = defaultdict(list)
    for r in kept:
        by_obj[r["object"]].append(r)
    jobs = []
    for obj, rows in sorted(by_obj.items()):
        rows.sort(key=lambda r: (r["setting"], Path(r["path"]).parent.name, Path(r["path"]).stem))
        jobs += [(r, f"{obj}_{i:03d}") for i, r in enumerate(rows, 1)]
    (DATA / "masks").mkdir(parents=True, exist_ok=True)
    manifest = []
    with ThreadPoolExecutor(6) as pool:
        for n, res in enumerate(pool.map(copy_one, jobs), 1):
            r = res["row"]
            folder = PHOTOS / r["object"]
            folder.mkdir(parents=True, exist_ok=True)
            # the number first, the setting after it: <object>_<NNN>_<white|lab>.jpg
            out = folder / f"{res['name']}.jpg"
            ok, buf = cv2.imencode(".jpg", res["photo"], [cv2.IMWRITE_JPEG_QUALITY, 92])
            buf.tofile(str(out))
            cv2.imwrite(str(DATA / "masks" / f"{res['name']}.png"), res["mask"])
            back = cv2.imdecode(np.fromfile(str(out), np.uint8), cv2.IMREAD_COLOR)
            if back is None or back.shape != res["photo"].shape:
                raise RuntimeError(f"written photo does not read back: {out}")
            manifest.append({"photo": f"museum_oltenia/{r['object']}/{out.name}", "object": r["object"],
                             "label": labels[r["object"]], "setting": res["name"].rsplit("_", 1)[1],
                             "source": str(Path(r["path"]).relative_to(SOURCE)).replace("\\", "/"),
                             "original_size": res["original_size"], "sharpness": round(r["lap"], 1)})
            if n % 500 == 0:
                print(f"{n}/{len(jobs)} photos copied", flush=True)
    with open(DATA / "photos.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(manifest[0]))
        writer.writeheader()
        writer.writerows(sorted(manifest, key=lambda m: m["photo"]))
    print(f"copied {len(manifest)} photos of {len(by_obj)} objects; manifest {DATA / 'photos.csv'}")


if __name__ == "__main__":
    main()
