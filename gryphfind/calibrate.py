"""Print raw CLIP similarities so you can tune gryphfind/scoring.py.

Usage:
  python -m gryphfind.calibrate photo1.jpg photo2.jpg "blue water bottle" ...

Take two photos of the SAME item and one of a DIFFERENT item, plus a description,
and set the img_img / cross midpoints in scoring.py between the same/different values.
"""
import sys
from pathlib import Path

from PIL import Image

from .embedder import get_embedder

emb = get_embedder()
items = []
for a in sys.argv[1:]:
    if Path(a).exists():
        items.append((Path(a).name, "img", emb.embed_image(Image.open(a))))
    else:
        items.append((a, "txt", emb.embed_text(a)))

for i, (na, ta, va) in enumerate(items):
    for nb, tb, vb in items[i + 1:]:
        kind = "img_img" if ta == tb == "img" else "txt_txt" if ta == tb == "txt" else "cross"
        print(f"{kind:8s} {float(va @ vb):.3f}  {na}  <->  {nb}")
