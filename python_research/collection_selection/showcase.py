"""The showcase of the Muzeul Olteniei collection: for every object, PER_OBJECT of its photos spread evenly
over its views (the photos are numbered white backdrop / laboratory, view after view) and each one's
harmonization in a hall. Writes data/museum_oltenia/showcase.csv (object, label, photo, harmonized:
paths relative to the project's root); results.html shows the first of each object, and the public
repository carries these files only (all 8,744 photos and their harmonizations stay on the laptop).

Usage: python collection_selection/showcase.py"""
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "python_research" / "data" / "museum_oltenia"
PER_OBJECT = 5


def main() -> None:
    photos = defaultdict(list)
    for m in csv.DictReader(open(DATA / "photos.csv", encoding="utf-8")):
        photos[m["object"]].append(m)
    rows, missing = [], 0
    for obj, items in sorted(photos.items()):
        items.sort(key=lambda m: m["photo"])
        for k in np.unique(np.linspace(0, len(items) - 1, min(PER_OBJECT, len(items))).round().astype(int)):
            m = items[k]
            stem = Path(m["photo"]).stem
            done = sorted((DATA / "harmonized" / obj).glob(f"{stem}_on_*.jpg"))
            if not done:
                missing += 1
                continue
            rows.append({"object": obj, "label": m["label"], "photo": f"photos/{m['photo']}",
                         "harmonized": done[0].relative_to(ROOT).as_posix()})
    with open(DATA / "showcase.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["object", "label", "photo", "harmonized"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"{len(rows)} photos of {len({r['object'] for r in rows})} objects; {missing} not harmonized yet")


if __name__ == "__main__":
    main()
