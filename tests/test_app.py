import io, os, tempfile
os.environ["EMBEDDER"] = "stub"
_tmp = tempfile.mkdtemp()
os.environ["GRYPHFIND_DB"] = os.path.join(_tmp, "t.db")
os.environ["GRYPHFIND_UPLOADS"] = os.path.join(_tmp, "up")
from PIL import Image
from gryphfind.app import app
from gryphfind.scoring import combine

def img(color):
    b = io.BytesIO(); Image.new("RGB", (64, 64), color).save(b, "PNG"); b.seek(0); return b

def post(c, **form):
    data = {k: v for k, v in form.items() if k != "image"}
    if "image" in form: data["image"] = (form["image"], "x.png")
    return c.post("/api/posts", data=data, content_type="multipart/form-data")

def test_flow():
    c = app.test_client()
    assert c.get("/").status_code == 200
    assert post(c, kind="found", title="Blue water bottle", image=img((20, 40, 200))).status_code == 201
    assert post(c, kind="found", title="Red umbrella", image=img((200, 20, 20))).status_code == 201
    r = post(c, kind="lost", title="blue water bottle", image=img((25, 45, 205)))
    assert r.status_code == 201
    m = r.get_json()["matches"]
    assert m[0]["post"]["title"] == "Blue water bottle"
    assert m[0]["score"] > m[1]["score"]
    assert "why" in m[0]
    # text-only lost post still gets results
    r2 = post(c, kind="lost", title="red umbrella")
    assert r2.get_json()["matches"][0]["post"]["title"] == "Red umbrella"
    # resolve
    lost_id, found_id = r.get_json()["post"]["id"], m[0]["post"]["id"]
    assert c.post(f"/api/posts/{lost_id}/resolve", json={"match_id": found_id}).status_code == 200
    st = c.get("/api/stats").get_json()
    assert st["reunited"] == 1 and st["open_found"] == 1
    assert c.get(f"/api/posts/{found_id}/matches").status_code == 200
    assert c.get("/api/posts/nope/matches").status_code == 404
    assert post(c, kind="lost").status_code == 400
    assert post(c, kind="banana", title="x").status_code == 400

def test_scoring():
    assert combine({"img_img": 0.92, "txt_txt": None, "cross": None})[0] > 90
    assert combine({"img_img": 0.55, "txt_txt": None, "cross": None})[0] < 5
    assert combine({"img_img": None, "txt_txt": None, "cross": 0.31})[0] > 90
    assert combine({})[0] == 0
