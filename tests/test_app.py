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
    # claim -> stays visible as "claimed"
    lost_id, found_id = r.get_json()["post"]["id"], m[0]["post"]["id"]
    cl = c.post("/api/claims", json={"a": lost_id, "b": found_id})
    assert cl.status_code == 201
    cid = cl.get_json()["claim"]["id"]
    board = c.get("/api/posts?kind=found").get_json()["posts"]
    assert any(p["id"] == found_id and p["status"] == "claimed" for p in board)
    assert c.post("/api/claims", json={"a": lost_id, "b": found_id}).status_code == 409  # double claim
    # wrong owner -> put back; that pair is never suggested again
    assert c.post(f"/api/claims/{cid}/reject").status_code == 200
    again = c.get(f"/api/posts/{lost_id}/matches").get_json()["matches"]
    assert all(x["post"]["id"] != found_id for x in again)
    assert all(p["status"] == "open" for p in c.get("/api/posts?kind=found").get_json()["posts"])
    # real owner claims and confirms handover
    real = post(c, kind="lost", title="blue water bottle", image=img((22, 42, 202))).get_json()
    cid2 = c.post("/api/claims", json={"a": real["post"]["id"], "b": found_id}).get_json()["claim"]["id"]
    assert c.post(f"/api/claims/{cid2}/confirm").status_code == 200
    st = c.get("/api/stats").get_json()
    assert st["reunited"] == 1 and st["open_found"] == 1
    hist = c.get("/api/claims").get_json()["claims"]
    assert {h["status"] for h in hist} == {"rejected", "confirmed"}
    # expiry: age the umbrella to 28 days -> expiring soon; to 31 -> gone
    umb = [p for p in c.get("/api/posts?kind=found").get_json()["posts"] if p["title"] == "Red umbrella"][0]
    assert umb["days_left"] == 30
    assert c.post(f"/api/demo/age/{umb['id']}", json={"days": 28}).get_json()["post"]["days_left"] == 2
    assert [p["id"] for p in c.get("/api/expiring").get_json()["posts"]] == [umb["id"]]
    assert c.get("/api/stats").get_json()["expiring_soon"] == 1
    c.post(f"/api/demo/age/{umb['id']}", json={"days": 3})
    assert c.get("/api/posts?kind=found").get_json()["posts"] == []
    assert c.get(f"/api/posts/{found_id}/matches").status_code == 200
    assert c.get("/api/posts/nope/matches").status_code == 404
    assert post(c, kind="lost").status_code == 400
    assert post(c, kind="banana", title="x").status_code == 400

def test_scoring():
    assert combine({"img_img": 0.92, "txt_txt": None, "cross": None})[0] > 90
    assert combine({"img_img": 0.55, "txt_txt": None, "cross": None})[0] < 5
    assert combine({"img_img": None, "txt_txt": None, "cross": 0.31})[0] > 90
    assert combine({})[0] == 0
