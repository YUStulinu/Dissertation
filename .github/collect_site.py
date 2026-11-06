"""The GitHub Pages site: index.html, results.html and every file they show or link to - not the whole
repository (over 1 GB with the live demo's object views, more than Pages publishes).
Usage: python3 .github/collect_site.py <output folder>"""
import re
import shutil
import sys
from pathlib import Path

root, out = Path(__file__).resolve().parents[1], Path(sys.argv[1])
files = {"index.html", "results.html", ".nojekyll"}
for page in ("index.html", "results.html"):
    files |= set(re.findall(r'(?:src|href)="([^"#:?]+)"', (root / page).read_text(encoding="utf-8")))
missing = sorted(f for f in files if not (root / f).is_file())
if missing:
    sys.exit(f"Missing files the page refers to: {missing}")
for f in sorted(files):
    (out / f).parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(root / f, out / f)
print(f"{len(files)} files, {sum((root / f).stat().st_size for f in files) / 2**20:.1f} MB")
