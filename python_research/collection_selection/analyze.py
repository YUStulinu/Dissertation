"""Quality analysis of every photo of the series (series.json): object box (YOLOE, text prompt of the
series), its SAM 2.1 mask, the turntable, people, sharpness. One JSON line per photo in
analysis/metrics.jsonl and the object mask (1/4 resolution) in analysis/masks/; resumable.

Run from python_research/models (YOLOE's text model lives there)."""
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, r"C:\MASTER\disertatie\python_research")
sys.stdout.reconfigure(encoding="utf-8")
HERE = Path(__file__).parent
WORK = Path(__file__).resolve().parents[1] / "data" / "museum_oltenia" / "analysis"  # metrics, masks, series.json, sizes.json
OUT = WORK
MODELS = Path(r"C:\MASTER\disertatie\python_research\models")
PEOPLE = ["person", "hand", "foot", "shoe"]
GENERIC = ["ancient artifact", "pottery", "figurine", "metal object"]  # besides the series' own prompt


def iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / max(1e-6, (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


def read_reduced(path: Path) -> np.ndarray:
    return cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_REDUCED_COLOR_4)


def sharpness(gray: np.ndarray, mask: np.ndarray) -> tuple[float, float]:
    """Variance of the Laplacian and mean Tenengrad inside the eroded object mask."""
    inner = cv2.erode(mask, np.ones((7, 7), np.uint8)) > 0
    if inner.sum() < 50:
        return 0.0, 0.0
    lap = cv2.Laplacian(gray, cv2.CV_32F)
    gx, gy = cv2.Sobel(gray, cv2.CV_32F, 1, 0), cv2.Sobel(gray, cv2.CV_32F, 0, 1)
    return float(lap[inner].var()), float(np.sqrt(gx * gx + gy * gy)[inner].mean())


def whiteness(image: np.ndarray, mask: np.ndarray) -> tuple[float, float]:
    """Share of white (bright, unsaturated) pixels in the whole mask and in its lowest 20% of rows:
    a white turntable taken into the mask makes the bottom far whiter than the object."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    white = (hsv[..., 1] < 32) & (hsv[..., 2] > 150)
    m = mask > 0
    ys = np.nonzero(m.any(1))[0]
    low = m.copy()
    low[:ys[0] + int(0.8 * (ys[-1] - ys[0] + 1))] = False
    return round(float(white[m].mean()), 4), round(float(white[low].mean()) if low.any() else 0.0, 4)


def first_mask(results) -> np.ndarray | None:
    r = results[0]
    if r.masks is None or not r.masks.data.shape[0]:
        return None
    m = (r.masks.data[0].cpu().numpy() > 0.5).astype(np.uint8) * 255
    return m if m.any() else None


def largest_part(mask: np.ndarray) -> np.ndarray:
    n, labels, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8))
    if n <= 2:
        return mask
    return np.where(labels == 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA])), 255, 0).astype(np.uint8)


def without_stand(sam, mask: np.ndarray):
    """The object's mask without the turntable it stands on. SAM is asked what lies just under the mask's
    bottom (the turntable's top or side, the floor of the photo) and that region is taken out of the mask,
    which SAM's box prompt sometimes extends over the turntable. -> (mask, share removed, stand mask)."""
    h, w = mask.shape
    ys, xs = np.nonzero(mask)
    bottom = ys.max()
    x = int(np.median(xs[ys >= bottom - max(2, 0.03 * (bottom - ys.min()))]))
    y = int(min(h - 2, bottom + max(3, 0.02 * h)))
    if y <= bottom:
        return mask, 0.0, None
    stand = first_mask(sam(points=[[x, y]], labels=[1]))
    if stand is None or stand.mean() / 255 > 0.7 or np.count_nonzero(stand & mask) > 0.5 * np.count_nonzero(mask):
        return mask, 0.0, stand  # nothing, the whole photo or the object itself: keep the mask as it is
    kept = largest_part(np.where(stand > 0, 0, mask).astype(np.uint8))
    return kept, 1.0 - np.count_nonzero(kept) / np.count_nonzero(mask), stand


def main() -> None:
    from ultralytics import YOLOE
    from ultralytics.models.sam import SAM2Predictor

    series = json.load(open(WORK / "series.json", encoding="utf-8"))
    (OUT / "masks").mkdir(parents=True, exist_ok=True)
    log = OUT / "metrics.jsonl"
    done = set()
    if log.exists():
        done = {json.loads(l)["path"] for l in open(log, encoding="utf-8")}
    yolo = YOLOE(str(MODELS / "yoloe-11l-seg.pt"))
    sam = SAM2Predictor(overrides=dict(conf=0.25, task="segment", mode="predict", imgsz=1024, save=False,
                                       verbose=False, model=str(MODELS / "sam2.1_b.pt")))
    total = sum(s["photos"] for s in series)
    n, t0 = len(done), time.time()
    with open(log, "a", encoding="utf-8") as out:
        for s in series:
            files = sorted(Path(s["folder"]).glob("*.jpg"))
            if all(str(f) in done for f in files):
                continue
            targets = [s["prompt"]] + [g for g in GENERIC if g != s["prompt"]]
            classes = targets + ["turntable"] + PEOPLE
            yolo.set_classes(classes)
            for f in files:
                if str(f) in done:
                    continue
                row = {"path": str(f), "object": s["object"], "setting": s["setting"]}
                try:
                    im = read_reduced(f)
                    h, w = im.shape[:2]
                    row["size"] = [w, h]
                    r = yolo(im, conf=0.05, verbose=False)[0]
                    dets = [(classes[int(c)], float(p), [float(v) for v in b])
                            for c, p, b in zip(r.boxes.cls, r.boxes.conf, r.boxes.xyxy)]
                    table = max((d for d in dets if d[0] == "turntable"), key=lambda d: d[1], default=None)
                    obj = max((d for d in dets if d[0] in targets and (table is None or iou(d[2], table[2]) < 0.6)),
                              key=lambda d: d[1] * (1.5 if d[0] == s["prompt"] else 1.0), default=None)
                    people = [d for d in dets if d[0] in PEOPLE and d[1] >= 0.25]
                    row["people"] = [[d[0], round(d[1], 3)] for d in people]
                    row["table"] = None if table is None else [round(table[1], 3)] + [round(v, 1) for v in table[2]]
                    if obj is None:
                        row["status"] = "no_object"
                    else:
                        row["det"] = [obj[0], round(obj[1], 3)] + [round(v, 1) for v in obj[2]]
                        sam.set_image(im)
                        m = first_mask(sam(bboxes=[obj[2]]))
                        if m is None:
                            row["status"] = "no_mask"
                        else:
                            m, cut, stand = without_stand(sam, m)
                            sam.reset_image()
                            row["stand_cut"] = round(cut, 4)
                            if stand is not None:
                                sy, sx = np.nonzero(stand)
                                row["stand_box"] = [int(sx.min()), int(sy.min()), int(sx.max()), int(sy.max())]
                            ys, xs = np.nonzero(m)
                            row["mask_box"] = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
                            row["area"] = round(float((m > 0).mean()), 5)
                            row["border"] = bool(m[:2].any() or m[-2:].any() or m[:, :2].any() or m[:, -2:].any())
                            row["lap"], row["ten"] = sharpness(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), m)
                            row["white_all"], row["white_bottom"] = whiteness(im, m)
                            cv2.imwrite(str(OUT / "masks" / f"{s['object']}__{s['setting']}__{f.stem}.png"), m)
                            row["status"] = "ok"
                except Exception as e:  # a broken file must not stop the run
                    row["status"] = f"error: {e}"
                out.write(json.dumps(row) + "\n")
                n += 1
                if n % 200 == 0:
                    out.flush()
                    rate = (n - len(done)) / (time.time() - t0)
                    print(f"{n}/{total} photos, {rate:.1f}/s, ~{(total - n) / max(rate, 1e-6) / 60:.0f} min left", flush=True)
    print("done", n)


if __name__ == "__main__":
    main()
