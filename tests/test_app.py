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

FOUND_KEYS = {}


def test_flow():
    c = app.test_client()
    assert c.get("/").status_code == 200
    for title, color in (("Blue water bottle", (20, 40, 200)), ("Red umbrella", (200, 20, 20))):
        res = post(c, kind="found", title=title, image=img(color))
        assert res.status_code == 201 and res.get_json()["key"]
        FOUND_KEYS[title] = res.get_json()["key"]
    r = post(c, kind="lost", title="blue water bottle", image=img((25, 45, 205)))
    assert r.status_code == 201
    m = r.get_json()["matches"]
    assert m[0]["post"]["title"] == "Blue water bottle"
    assert m[0]["score"] > m[1]["score"]
    assert "why" in m[0]
    # text-only lost post still gets results
    r2 = post(c, kind="lost", title="red umbrella")
    assert r2.get_json()["matches"][0]["post"]["title"] == "Red umbrella"
    # --- roles ---------------------------------------------------------
    finder_key = None
    found_posts = c.get("/api/posts?kind=found").get_json()["posts"]
    found_id = [p for p in found_posts if p["title"] == "Blue water bottle"][0]["id"]
    lost_id, lost_key = r.get_json()["post"]["id"], r.get_json()["key"]
    # a stranger can't claim using someone else's lost post
    assert c.post("/api/claims", json={"lost_id": lost_id, "found_id": found_id, "key": "nope", "proof": "black lid"}).status_code == 409
    # proof is required
    assert c.post("/api/claims", json={"lost_id": lost_id, "found_id": found_id, "key": lost_key, "proof": ""}).status_code == 409
    # real owner requests; a "wrong" person also requests the same bottle
    req = c.post("/api/claims", json={"lost_id": lost_id, "found_id": found_id, "key": lost_key, "proof": "Gryphons sticker under the lid"})
    assert req.status_code == 201
    good_cid = req.get_json()["claim"]["id"]
    assert c.post("/api/claims", json={"lost_id": lost_id, "found_id": found_id, "key": lost_key, "proof": "again"}).status_code == 409
    wrong = post(c, kind="lost", title="blue bottle", image=img((24, 44, 204))).get_json()
    bad_cid = c.post("/api/claims", json={"lost_id": wrong["post"]["id"], "found_id": found_id,
                                          "key": wrong["key"], "proof": "it's blue"}).get_json()["claim"]["id"]
    # item stays on the board with 2 requests waiting
    assert c.get("/api/claims").get_json()["requests"][found_id] == 2
    assert any(p["id"] == found_id for p in c.get("/api/posts?kind=found").get_json()["posts"])
    # public history never leaks proofs
    assert "proof" not in str(c.get("/api/claims").get_json())
    # only the finder can decide: lost person and strangers are refused
    assert c.post(f"/api/claims/{good_cid}/hand-over", json={"key": lost_key}).status_code == 403
    assert c.post(f"/api/claims/{bad_cid}/decline", json={"key": "nope"}).status_code == 403
    # the finder's key came back when the found post was made
    finder_key = FOUND_KEYS["Blue water bottle"]
    inbox = c.post("/api/mine", json={"keys": {found_id: finder_key}}).get_json()["posts"]
    assert len(inbox) == 1 and len(inbox[0]["requests"]) == 2
    assert {q["proof"] for q in inbox[0]["requests"]} == {"Gryphons sticker under the lid", "it's blue"}
    # the lost person sees status only, no proofs of others
    mine = c.post("/api/mine", json={"keys": {lost_id: lost_key}}).get_json()["posts"]
    assert mine[0]["requests"][0]["status"] == "pending" and "proof" not in mine[0]["requests"][0]
    # finder hands it to the right person -> the other request is declined automatically
    assert c.post(f"/api/claims/{good_cid}/hand-over", json={"key": finder_key}).status_code == 200
    statuses = {h["id"]: h["status"] for h in c.get("/api/claims").get_json()["claims"]}
    assert statuses[good_cid] == "confirmed" and statuses[bad_cid] == "declined"
    assert all(p["id"] != found_id for p in c.get("/api/posts?kind=found").get_json()["posts"])
    assert c.post(f"/api/claims/{good_cid}/decline", json={"key": finder_key}).status_code == 403  # already answered
    st = c.get("/api/stats").get_json()
    assert st["reunited"] == 1 and st["open_found"] == 1
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
