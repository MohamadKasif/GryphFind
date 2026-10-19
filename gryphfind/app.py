"""GryphFind web server."""
from __future__ import annotations

import io
import os
import uuid
from pathlib import Path

from flask import Flask, abort, jsonify, request, send_from_directory
from PIL import Image, ImageOps

from .embedder import CATEGORIES, get_embedder
from .explain import explain
from .store import EXPIRY_DAYS, OPPOSITE, ClaimError, Post, get_store

ROOT = Path(__file__).resolve().parent
UPLOADS = Path(os.getenv("GRYPHFIND_UPLOADS", ROOT.parent / "uploads"))
UPLOADS.mkdir(parents=True, exist_ok=True)
EXPLAIN_TOP = int(os.getenv("EXPLAIN_TOP", "3"))
DEMO = os.getenv("GRYPHFIND_DEMO", "1") == "1"  # shows the fast-forward button for judging

app = Flask(__name__, static_folder=str(ROOT / "static"), static_url_path="/static")
app.config["MAX_CONTENT_LENGTH"] = 12 * 1024 * 1024

store = get_store()
emb = get_embedder()


def _save_image(file) -> tuple[str, Image.Image]:
    img = ImageOps.exif_transpose(Image.open(io.BytesIO(file.read()))).convert("RGB")
    img.thumbnail((1024, 1024))
    name = f"{uuid.uuid4().hex[:16]}.jpg"
    img.save(UPLOADS / name, "JPEG", quality=85)
    return name, img


def _with_explanations(p: Post, matches: list[dict]) -> list[dict]:
    for m in matches[:EXPLAIN_TOP]:
        if m["score"] >= 35:
            other = store.get(m["post"]["id"])
            if other:
                m["why"] = explain(store, p, other, m["score"], m["signals"])
    return matches


@app.get("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.get("/uploads/<path:name>")
def uploads(name):
    return send_from_directory(UPLOADS, name)


BRAND = ROOT / "static" / "brand"


def _brand_file(stem: str):
    """Use an official logo/mascot image if the team dropped one into static/brand/."""
    for ext in ("svg", "png", "webp", "jpg", "jpeg"):
        f = BRAND / f"{stem}.{ext}"
        if f.exists():
            return f"/static/brand/{f.name}?v={int(f.stat().st_mtime)}"
    return None


@app.get("/api/meta")
def meta():
    return jsonify(categories=CATEGORIES, backend=store.backend, embedder=emb.name,
                   expiry_days=EXPIRY_DAYS, demo=DEMO,
                   brand={"logo": _brand_file("logo"), "mascot": _brand_file("mascot")})


@app.post("/api/posts")
def create_post():
    f = request.form
    kind = f.get("kind")
    if kind not in OPPOSITE:
        abort(400, "kind must be 'lost' or 'found'")
    title = (f.get("title") or "").strip()
    desc = (f.get("description") or "").strip()
    file = request.files.get("image")
    if not title and not desc and not file:
        abort(400, "add a photo or a description")

    p = Post(kind=kind, title=title[:120], description=desc[:1000],
             location=(f.get("location") or "").strip()[:80],
             contact=(f.get("contact") or "").strip()[:120])

    if file and file.filename:
        p.image_path, img = _save_image(file)
        p.img_vec = emb.embed_image(img)
    text = ". ".join(x for x in (title, desc) if x)
    if text:
        p.txt_vec = emb.embed_text(text)

    chosen = f.get("category") or ""
    if chosen in CATEGORIES:
        p.category = chosen
    elif p.img_vec is not None:
        p.category = emb.classify(p.img_vec)
    else:
        p.category = "other"

    if not p.title:
        p.title = f"{kind.title()} {p.category}"

    store.add(p)
    return jsonify(post=p.public(), matches=_with_explanations(p, store.matches(p))), 201


@app.get("/api/posts")
def list_posts():
    kind = request.args.get("kind")
    return jsonify(posts=[p.public() for p in store.list(kind if kind in OPPOSITE else None)])


@app.get("/api/posts/<pid>/matches")
def post_matches(pid):
    p = store.get(pid)
    if not p:
        abort(404)
    return jsonify(post=p.public(), matches=_with_explanations(p, store.matches(p)))


@app.get("/api/claims")
def list_claims():
    return jsonify(claims=store.claims(), pending=store.pending_claim_ids())


@app.post("/api/claims")
def make_claim():
    body = request.get_json(silent=True) or {}
    try:
        return jsonify(claim=store.claim(body.get("a", ""), body.get("b", ""))), 201
    except ClaimError as e:
        abort(409, str(e))


@app.post("/api/claims/<cid>/confirm")
def confirm_claim(cid):
    try:
        store.confirm(cid)
    except ClaimError as e:
        abort(409, str(e))
    return jsonify(ok=True)


@app.post("/api/claims/<cid>/reject")
def reject_claim(cid):
    try:
        store.reject(cid)
    except ClaimError as e:
        abort(409, str(e))
    return jsonify(ok=True)


@app.get("/api/expiring")
def expiring():
    days = min(float(request.args.get("days", 3)), EXPIRY_DAYS)
    return jsonify(posts=[p.public() for p in store.expiring(days)], days=days)


@app.post("/api/demo/age/<pid>")
def demo_age(pid):
    if not DEMO:
        abort(404)
    if not store.get(pid):
        abort(404)
    days = float((request.get_json(silent=True) or {}).get("days", EXPIRY_DAYS - 2))
    store.age(pid, max(0.0, min(days, 365.0)))
    return jsonify(post=store.get(pid).public())


@app.get("/api/stats")
def stats():
    return jsonify(store.stats())


@app.errorhandler(400)
@app.errorhandler(404)
@app.errorhandler(409)
def err(e):
    return jsonify(error=getattr(e, "description", str(e))), e.code


def main():
    app.run(host=os.getenv("HOST", "0.0.0.0"), port=int(os.getenv("PORT", "8000")), debug=False)


if __name__ == "__main__":
    main()
