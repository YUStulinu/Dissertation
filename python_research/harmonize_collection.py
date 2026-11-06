"""
Module 2 on the Muzeul Olteniei Craiova collection: every selected photo of photos/museum_oltenia/
is cut out and placed in a museum hall with the unified pipeline's steps - matted cut-out from the
photo's mask (data/museum_oltenia/masks/), ensemble harmonization, camera match, a faint floor
reflection and a contact shadow - like museum_demo.py does for the Met objects.

The objects are 3-25 cm tall: at their real size on the floor of a hall photographed from several
metres they would be a few dozen pixels high, too small to show anything. Each is therefore shown as
a display piece, its height a fixed share of the hall photo (less for wide objects so that they fit),
on a free spot of the floor in front of the hall's exhibits (HALLS); the halls take turns. The
estimated real sizes are listed in data/museum_oltenia/objects.csv.

Saves data/museum_oltenia/harmonized/<object>/<photo>_on_<hall>.jpg (photos already done are
skipped, so an interrupted run continues) and, with --sheet, one contact sheet per object.

Usage:
    python harmonize_collection.py                  # all photos not harmonized yet
    python harmonize_collection.py --object jug_I8276 --sheet
"""

import argparse
import time
from pathlib import Path

import cv2
import numpy as np

from common.composition import add_reflection, add_shadow, dilate, inpaint, place
from common.harmonization import harmonize, match_camera
from common.imaging import labeled_grid, read_image, read_mask, write_image
from common.paths import DATA_DIR, IMAGES_DIR
from common.scene import depth_map, floor_foreshortening
from common.segmentation import extract_foreground

PHOTOS_DIR = IMAGES_DIR / "museum_oltenia"
COLLECTION_DIR = DATA_DIR / "museum_oltenia"
MASKS_DIR = COLLECTION_DIR / "masks"
OUT_DIR = COLLECTION_DIR / "harmonized"
# per hall: free spots of floor (bottom-centre of the object, fractions of the hall photo), each nearer to
# the camera than the exhibits around it so that the object may cover them; the object's height as a share
# of the hall's height; and the largest share of the hall's width it may take (a sword is made smaller)
HALLS = {
    "hall_kielce": ([(0.38, 0.9), (0.52, 0.88), (0.64, 0.92)], 0.32, 0.4),
    "hall_met_sculpture_court": ([(0.87, 0.9), (0.1, 0.92)], 0.26, 0.16),
    "hall_nelson_atkins": ([(0.6, 0.88), (0.7, 0.95), (0.53, 0.98)], 0.3, 0.3),
    "hall_norton_simon": ([(0.3, 0.88), (0.74, 0.86), (0.86, 0.95)], 0.3, 0.25),
}
MATTING_PX = 1600  # as build_catalog.py: the matting's soft edge of a 1600 px photo


def cut_out(photo: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """The matted BGRA cut-out of the object, its edge colours cleaned of the photo's background."""
    background = inpaint(photo, dilate(mask, 7), "telea")
    return extract_foreground(photo, mask, background, pad=2,
                              radius=max(2, round(0.004 * min(MATTING_PX, max(photo.shape[:2])))))


def display_scale(sprite: np.ndarray, hall: np.ndarray, share: float, max_width: float) -> float:
    h, w = sprite.shape[:2]
    return min(share, max_width * hall.shape[1] / hall.shape[0] * h / w)


def compose(sprite: np.ndarray, hall: np.ndarray, depth: np.ndarray, pos: tuple[float, float],
            share: float, max_width: float) -> tuple[np.ndarray, np.ndarray]:
    """-> (final image, the object's alpha in it)."""
    composed, alpha = place(sprite, hall, display_scale(sprite, hall, share, max_width), pos)
    harmonized = match_camera(harmonize(composed, alpha, "ensemble"), alpha)
    anchor = (min(round(pos[0] * hall.shape[1]), hall.shape[1] - 1), min(round(pos[1] * hall.shape[0]), hall.shape[0] - 1))
    final = add_shadow(add_reflection(harmonized, alpha, 0.2), alpha, "ellipse", tilt=floor_foreshortening(depth, anchor))
    return final, alpha


def photos(only: str | None = None) -> list[Path]:
    return sorted(p for p in PHOTOS_DIR.glob("*/*.jpg") if only in (None, p.parent.name))


def contact_sheet(obj: str) -> Path | None:
    results = sorted((OUT_DIR / obj).glob("*.jpg"))
    if not results:
        return None
    picks = [results[int(i)] for i in np.linspace(0, len(results) - 1, min(12, len(results)))]
    grid = labeled_grid([(read_image(p), p.stem.split("_on_")[0]) for p in picks], cols=4, height=360)
    return write_image(COLLECTION_DIR / "sheets" / f"{obj}.jpg", grid)


def main() -> None:
    parser = argparse.ArgumentParser(description="Harmonize the Muzeul Olteniei photos into museum halls")
    parser.add_argument("--object", help="Only this object (its folder name in photos/museum_oltenia)")
    parser.add_argument("--sheet", action="store_true", help="Also save a contact sheet per object")
    args = parser.parse_args()

    halls = {name: read_image(IMAGES_DIR / f"{name}.jpg") for name in HALLS}
    depths = {name: depth_map(img, "indoor") for name, img in halls.items()}
    todo = photos(args.object)
    t0, done = time.time(), 0
    names = list(HALLS)
    for i, path in enumerate(todo):
        hall_name = names[i % len(names)]
        places, share, max_width = HALLS[hall_name]
        out = OUT_DIR / path.parent.name / f"{path.stem}_on_{hall_name}.jpg"
        if any((OUT_DIR / path.parent.name).glob(f"{path.stem}_on_*.jpg")):  # done, in whichever hall
            continue
        mask_file = MASKS_DIR / f"{path.stem}.png"
        if not mask_file.exists():
            print(f"  no mask for {path.name}, skipped")
            continue
        photo = read_image(path)
        sprite = cut_out(photo, read_mask(mask_file, photo.shape[:2]))
        final, _ = compose(sprite, halls[hall_name], depths[hall_name], places[(i // len(names)) % len(places)],
                           share, max_width)
        write_image(out, final)
        done += 1
        if done % 50 == 0:
            rate = done / (time.time() - t0)
            print(f"{i + 1}/{len(todo)} photos, {rate:.2f}/s, ~{(len(todo) - i - 1) / rate / 3600:.1f} h left", flush=True)
    print(f"harmonized {done} photos")
    if args.sheet:
        for obj in sorted({p.parent.name for p in todo}):
            sheet = contact_sheet(obj)
            if sheet:
                print("Saved:", sheet)


if __name__ == "__main__":
    main()
