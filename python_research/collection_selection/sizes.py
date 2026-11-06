"""Estimated real size of every object, with the turntable it stands on as the ruler.

In each kept photo the turntable's width (the SAM region under the object, analysis "stand_box") and the
object's size are measured in pixels. Upright objects: their height in side views (the turntable seen
low, box height / width < 0.45). Flat objects: their longest side in views from above (> 0.7).
Per object and turntable kind (black disc or white cylinder, from the colour inside the turntable's box)
the median ratio object / turntable is taken. The two turntables' diameters come from the objects whose
heights the museum's tables give (black disc) and from the objects photographed on both (white cylinder).
Writes sizes.json: {object: {"height_m" or "length_m", "n", "turntable"}}."""
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).parent
WORK = Path(__file__).resolve().parents[1] / "data" / "museum_oltenia" / "analysis"  # metrics, masks, series.json, sizes.json
sys.path.insert(0, str(HERE))
from selection import _mask, load, select  # noqa: E402

SIDE_DROP = 0.2  # side view: the object's bottom at most this share of the turntable's width below its top edge
TOP_ROUND = 0.85  # view from above: the turntable's box nearly square
KNOWN = {"cup_I56404": 0.097, "bowl_I56651": 0.085}


def measure(r, shape):
    if "stand_box" not in r:
        return None
    w, h = r["size"]
    sx1, sy1, sx2, sy2 = r["stand_box"]
    if sx1 <= 2 or sx2 >= w - 3:  # the turntable cut by the photo's side: its width is unknown
        return None
    sw, sh = sx2 - sx1, sy2 - sy1
    x1, y1, x2, y2 = r["mask_box"]
    if not (1.2 * (x2 - x1) < sw):  # the region under the object is not wider than it: not the turntable
        return None
    elevation = sh / sw
    im = cv2.imdecode(np.fromfile(r["path"], np.uint8), cv2.IMREAD_REDUCED_GRAYSCALE_4)
    inside = np.zeros(im.shape, bool)
    inside[sy1:sy2, sx1:sx2] = True
    inside &= _mask(r) == 0
    kind = "black" if np.median(im[inside]) < 110 else "white"
    if shape == "upright" and (y2 - sy1) / sw < SIDE_DROP:
        return kind, (y2 - y1) / sw
    if shape == "flat" and elevation > TOP_ROUND:
        m = _mask(r) > 0
        ys, xs = np.nonzero(m)
        (_, _), (a, b), _ = cv2.minAreaRect(np.column_stack([xs, ys]).astype(np.float32))
        return kind, max(a, b) / sw
    return None


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    series = json.load(open(WORK / "series.json", encoding="utf-8"))
    shapes = {s["object"]: s["shape"] for s in series}
    ratios = defaultdict(lambda: defaultdict(list))
    for r in select(load(), shapes):
        if r["keep"]:
            m = measure(r, shapes[r["object"]])
            if m:
                ratios[r["object"]][m[0]].append(m[1])
    med = {o: {k: (float(np.median(v)), len(v)) for k, v in d.items() if len(v) >= 5} for o, d in ratios.items()}
    cal = {o: round(KNOWN[o] / med[o]["white"][0], 3) for o in KNOWN if "white" in med.get(o, {})}
    print("white cylinder diameter from", cal)
    d_white = float(np.mean(list(cal.values())))
    both = [med[o]["white"][0] / med[o]["black"][0] for o in med if "black" in med[o] and "white" in med[o]]
    d_black = d_white * float(np.median(both))
    print(f"black disc {d_black:.3f} m, white cylinder {d_white:.3f} m (from {len(both)} objects on both; spread "
          f"{np.percentile(both, 25):.2f}-{np.percentile(both, 75):.2f})")
    diam = {"black": d_black, "white": d_white}
    out = {}
    for o in sorted(shapes):
        est = [(med[o][k][0] * diam[k], med[o][k][1], k) for k in med.get(o, {})]
        if not est:
            print(f"{o:32s} no measurement")
            continue
        size = float(np.average([e[0] for e in est], weights=[e[1] for e in est]))
        key = "height_m" if shapes[o] == "upright" else "length_m"
        out[o] = {key: round(size, 3), "n": sum(e[1] for e in est), "turntable": "+".join(e[2] for e in est)}
        print(f"{o:32s} {key} {size * 100:5.1f} cm  " + "  ".join(f"{k}:{s * 100:.1f}cm(n={n})" for s, n, k in est))
    json.dump(out, open(WORK / "sizes.json", "w"), indent=1)


if __name__ == "__main__":
    main()
