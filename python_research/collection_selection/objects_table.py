"""Writes python_research/data/museum_oltenia/objects.csv: one row per object of the collection."""
import csv, json, re
from collections import Counter, defaultdict
from pathlib import Path
HERE = Path(__file__).parent
WORK = Path(__file__).resolve().parents[1] / "data" / "museum_oltenia" / "analysis"  # metrics, masks, series.json, sizes.json
DATA = Path(r"C:\MASTER\disertatie\python_research\data\museum_oltenia")
series = json.load(open(WORK / "series.json", encoding="utf-8"))
sizes = json.load(open(WORK / "sizes.json"))
known = {"cup_I56404": 0.097, "bowl_I56651": 0.085}
photos = list(csv.DictReader(open(DATA / "photos.csv", encoding="utf-8")))
kept = Counter(p["object"] for p in photos)
settings = defaultdict(Counter)
for p in photos: settings[p["object"]][p["setting"]] += 1
info = {}
for s in series:
    o = info.setdefault(s["object"], {"label": s["label"], "shape": s["shape"], "folders": set(), "taken": 0})
    o["folders"].add(Path(s["folder"]).parent.name); o["taken"] += s["photos"]
rows = []
for obj, o in sorted(info.items()):
    inv = re.search(r"(I|M|X|III)?_?([0-9][0-9_A-Z]*)$", obj)
    size = sizes.get(obj, {})
    if obj in known:
        value, how = known[obj], "museum table"
    elif size:
        value, how = size.get("height_m", size.get("length_m")), f"estimated from {size['n']} photos ({size['turntable']} turntable)"
    else:
        value, how = "", "not measured (turntable not fully in the photos)"
    rows.append({"object": obj, "label": o["label"], "museum_folder": " | ".join(sorted(o["folders"])),
                 "photos_taken": o["taken"], "photos_kept": kept[obj], "white_backdrop": settings[obj]["white"],
                 "lab": settings[obj]["lab"], "size_kind": "height" if o["shape"] == "upright" else "length",
                 "size_m": value, "size_source": how})
with open(DATA / "objects.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print(len(rows), "objects;", sum(1 for r in rows if r["size_m"] != ""), "with a size")
