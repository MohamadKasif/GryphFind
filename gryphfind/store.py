"""Storage + vector matching.

LocalStore     - SQLite file, similarity computed in numpy. Zero setup.
SnowflakeStore - posts live in a Snowflake table with VECTOR(FLOAT, 512) columns and
                 similarity is computed in the warehouse with VECTOR_COSINE_SIMILARITY.

Shared rules (BaseStore):
- Posts expire EXPIRY_DAYS (default 30) after they're created and drop off the board.
- Every post gets a secret owner key (only its SHA-256 hash is stored). The browser
  that made the post keeps the key; that's how we know who the finder is.
- The person who lost something sends a *claim request* to a found item, with a
  private proof only the finder sees. Several people can request the same item.
- Only the finder (holder of the found post's key) can hand it over to one
  requester (everything else on that item is declined) or decline a request.
  Declined pairs are never suggested again.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import math
import secrets
import os
import sqlite3
import time
import uuid
from dataclasses import asdict, dataclass, field

import numpy as np

from .embedder import DIM
from .embedder import get_semantic
from .scoring import MIN_MATCH, combine

OPPOSITE = {"lost": "found", "found": "lost"}
EXPIRY_DAYS = float(os.getenv("EXPIRY_DAYS", "30"))
DAY = 86400
VISIBLE = ("open", "claimed")


def cutoff() -> float:
    return time.time() - EXPIRY_DAYS * DAY


@dataclass
class Post:
    kind: str  # "lost" | "found"
    title: str
    description: str = ""
    location: str = ""
    contact: str = ""
    category: str = ""
    image_path: str = ""
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    created_at: float = field(default_factory=time.time)
    status: str = "open"  # open | claimed | resolved
    img_vec: np.ndarray | None = None
    txt_vec: np.ndarray | None = None

    def public(self) -> dict:
        d = asdict(self)
        d.pop("img_vec"), d.pop("txt_vec")
        d["image_url"] = f"/uploads/{self.image_path}" if self.image_path else None
        expires = self.created_at + EXPIRY_DAYS * DAY
        d["expires_at"] = expires
        d["days_left"] = max(0, math.ceil((expires - time.time()) / DAY))
        return d


def _sims(q: Post, c_img, c_txt) -> dict:
    def dot(a, b):
        return None if a is None or b is None else float(np.dot(a, b))

    cross = [s for s in (dot(q.txt_vec, c_img), dot(q.img_vec, c_txt)) if s is not None]
    return {
        "img_img": dot(q.img_vec, c_img),
        "txt_txt": dot(q.txt_vec, c_txt),
        "cross": max(cross) if cross else None,
    }


def _text(p) -> str:
    return ". ".join(x for x in (p.title, p.description) if x)


def _with_sem(sims: dict, q, c) -> dict:
    """Add the MiniLM meaning score for the two descriptions, when available."""
    sem = get_semantic()
    if sem is not None:
        sims["sem"] = sem.similarity(_text(q), _text(c))
    return sims


def _rank(rows, k):
    """Best first, and only the ones worth showing."""
    rows = [r for r in rows if r["score"] >= MIN_MATCH]
    rows.sort(key=lambda r: r["score"], reverse=True)
    return rows[:k]


class ClaimError(Exception):
    pass


class BaseStore:
    """Claims, expiry and stats, written once for both backends.
    Subclasses provide _q(sql, args) -> list[dict] with lowercase keys, _x(sql, args)
    for writes, P (the placeholder) and _row(dict) -> Post."""

    P = "?"
    LIST_COLS = "*"

    def _ph(self, n):
        return ",".join([self.P] * n)

    # ---- posts ----
    def get(self, pid: str) -> Post | None:
        rows = self._q(f"SELECT * FROM posts WHERE id={self.P}", (pid,))
        return self._row(rows[0]) if rows else None

    def list(self, kind: str | None = None, statuses=VISIBLE) -> list[Post]:
        q = (f"SELECT {self.LIST_COLS} FROM posts WHERE status IN ({self._ph(len(statuses))}) "
             f"AND created_at >= {self.P}")
        args = [*statuses, cutoff()]
        if kind:
            q += f" AND kind={self.P}"
            args.append(kind)
        return [self._row(r) for r in self._q(q + " ORDER BY created_at DESC", args)]

    def expiring(self, within_days: float = 3) -> list[Post]:
        lo, hi = cutoff(), cutoff() + within_days * DAY
        rows = self._q(
            f"SELECT {self.LIST_COLS} FROM posts WHERE status IN ({self._ph(len(VISIBLE))}) "
            f"AND created_at >= {self.P} AND created_at < {self.P} ORDER BY created_at ASC",
            [*VISIBLE, lo, hi],
        )
        return [self._row(r) for r in rows]

    def age(self, pid: str, days: float) -> None:
        """Demo helper: pretend a post was created `days` earlier."""
        self._x(f"UPDATE posts SET created_at = created_at - {self.P} WHERE id={self.P}", (days * DAY, pid))

    def rejected_pairs(self, pid: str) -> set[str]:
        rows = self._q(
            f"SELECT lost_id, found_id FROM claims WHERE status IN ('rejected','declined') "
            f"AND (lost_id={self.P} OR found_id={self.P})", (pid, pid))
        return {r["found_id"] if r["lost_id"] == pid else r["lost_id"] for r in rows}

    # ---- ownership ----
    @staticmethod
    def _hash(key: str) -> str:
        return hashlib.sha256(key.encode()).hexdigest()

    def new_key(self, pid: str) -> str:
        key = secrets.token_urlsafe(18)
        self._x(f"INSERT INTO post_keys (post_id, key_hash) VALUES ({self._ph(2)})", (pid, self._hash(key)))
        return key

    def owns(self, pid: str, key: str | None) -> bool:
        if not key:
            return False
        rows = self._q(f"SELECT key_hash FROM post_keys WHERE post_id={self.P}", (pid,))
        return bool(rows) and hmac.compare_digest(rows[0]["key_hash"], self._hash(key))

    # ---- claims ----
    def _claim(self, cid: str) -> dict:
        rows = self._q(f"SELECT * FROM claims WHERE id={self.P}", (cid,))
        if not rows:
            raise ClaimError("That claim doesn't exist.")
        return rows[0]

    def request(self, lost_id: str, found_id: str, key: str, proof: str) -> dict:
        """The person who lost something asks the finder for an item."""
        lost, found = self.get(lost_id), self.get(found_id)
        if not lost or not found or lost.kind != "lost" or found.kind != "found":
            raise ClaimError("Pick your lost post and a found item.")
        if not self.owns(lost.id, key):
            raise ClaimError("Only the person who posted the lost item can claim from this device.")
        if found.status == "resolved" or lost.status == "resolved":
            raise ClaimError("This item has already been handed over.")
        if found.id in self.rejected_pairs(lost.id):
            raise ClaimError("The finder already said this isn't a match.")
        if self._q(f"SELECT id FROM claims WHERE status='pending' AND lost_id={self.P} AND found_id={self.P}",
                   (lost.id, found.id)):
            raise ClaimError("You've already asked the finder about this item.")
        proof = (proof or "").strip()[:500]
        if len(proof) < 3:
            raise ClaimError("Add a detail only the owner would know, so the finder can check it's yours.")
        cid = uuid.uuid4().hex[:12]
        self._x(f"INSERT INTO claims (id, lost_id, found_id, created_at, status) VALUES ({self._ph(5)})",
                (cid, lost.id, found.id, time.time(), "pending"))
        self._x(f"INSERT INTO claim_proofs (claim_id, proof) VALUES ({self._ph(2)})", (cid, proof))
        return {"id": cid, "lost_id": lost.id, "found_id": found.id, "status": "pending"}

    def _finder_check(self, cid: str, key: str) -> dict:
        c = self._claim(cid)
        if not self.owns(c["found_id"], key):
            raise ClaimError("Only the person who found this item can decide.")
        if c["status"] != "pending":
            raise ClaimError("This request has already been answered.")
        return c

    def hand_over(self, cid: str, key: str) -> None:
        """Finder confirms the owner. Every other open request on either post is declined."""
        c = self._finder_check(cid, key)
        now = time.time()
        self._x(f"UPDATE claims SET status='declined', updated_at={self.P} WHERE status='pending' "
                f"AND id<>{self.P} AND (found_id={self.P} OR lost_id={self.P})",
                (now, cid, c["found_id"], c["lost_id"]))
        self._x(f"UPDATE claims SET status='confirmed', updated_at={self.P} WHERE id={self.P}", (now, cid))
        self._x(f"UPDATE posts SET status='resolved' WHERE id IN ({self._ph(2)})", (c["lost_id"], c["found_id"]))

    def decline(self, cid: str, key: str) -> None:
        self._finder_check(cid, key)
        self._x(f"UPDATE claims SET status='declined', updated_at={self.P} WHERE id={self.P}", (time.time(), cid))

    def claims(self, limit: int = 50) -> list[dict]:
        """Public history. Never includes proofs."""
        out = []
        for c in self._q(f"SELECT * FROM claims ORDER BY created_at DESC LIMIT {int(limit)}", ()):
            lost, found = self.get(c["lost_id"]), self.get(c["found_id"])
            if lost and found:
                out.append({**{k: c.get(k) for k in ("id", "status", "created_at", "updated_at")},
                            "lost": lost.public(), "found": found.public()})
        return out

    def pending_counts(self) -> dict[str, int]:
        """found post id -> number of open claim requests (for board badges)."""
        return {r["found_id"]: r["n"] for r in self._q(
            "SELECT found_id, COUNT(*) AS n FROM claims WHERE status='pending' GROUP BY found_id", ())}

    def mine(self, keys: dict[str, str]) -> list[dict]:
        """Posts this device owns, with the claim requests that concern them.
        Finders see each requester's proof and contact; requesters see status only."""
        out = []
        for pid, key in list(keys.items())[:100]:
            p = self.get(str(pid))
            if not p or not self.owns(p.id, str(key)):
                continue
            col = "found_id" if p.kind == "found" else "lost_id"
            reqs = []
            for c in self._q(f"SELECT * FROM claims WHERE {col}={self.P} ORDER BY created_at DESC", (p.id,)):
                other = self.get(c["lost_id"] if p.kind == "found" else c["found_id"])
                if not other:
                    continue
                r = {"id": c["id"], "status": c["status"], "created_at": c["created_at"], "other": other.public()}
                if p.kind == "found":
                    pr = self._q(f"SELECT proof FROM claim_proofs WHERE claim_id={self.P}", (c["id"],))
                    r["proof"] = pr[0]["proof"] if pr else ""
                reqs.append(r)
            out.append({"post": p.public(), "requests": reqs})
        out.sort(key=lambda x: x["post"]["created_at"], reverse=True)
        return out

    # ---- stats ----
    def stats(self) -> dict:
        one = lambda sql, args=(): list(self._q(sql, args)[0].values())[0]
        vis = f"status IN ({self._ph(len(VISIBLE))}) AND created_at >= {self.P}"
        args = (*VISIBLE, cutoff())
        return {
            "open_lost": one(f"SELECT COUNT(*) AS n FROM posts WHERE kind='lost' AND {vis}", args),
            "open_found": one(f"SELECT COUNT(*) AS n FROM posts WHERE kind='found' AND {vis}", args),
            "reunited": one("SELECT COUNT(*) AS n FROM claims WHERE status='confirmed'"),
            "pending_claims": one("SELECT COUNT(*) AS n FROM claims WHERE status='pending'"),
            "expiring_soon": len(self.expiring(3)),
            "expiry_days": EXPIRY_DAYS,
            "by_category": self._q(
                f"SELECT category, COUNT(*) AS n FROM posts WHERE {vis} GROUP BY category ORDER BY n DESC LIMIT 8", args),
            "by_location": self._q(
                f"SELECT location, COUNT(*) AS n FROM posts WHERE location<>'' GROUP BY location ORDER BY n DESC LIMIT 8", ()),
        }


class LocalStore(BaseStore):
    backend = "sqlite"
    P = "?"

    def __init__(self, path: str = "gryphfind.db"):
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.executescript(
            """CREATE TABLE IF NOT EXISTS posts (
                id TEXT PRIMARY KEY, kind TEXT, title TEXT, description TEXT,
                location TEXT, contact TEXT, category TEXT, image_path TEXT,
                created_at REAL, status TEXT, img_vec BLOB, txt_vec BLOB);
               CREATE TABLE IF NOT EXISTS claims (
                id TEXT PRIMARY KEY, lost_id TEXT, found_id TEXT,
                created_at REAL, updated_at REAL, status TEXT);
               CREATE TABLE IF NOT EXISTS post_keys (post_id TEXT PRIMARY KEY, key_hash TEXT);
               CREATE TABLE IF NOT EXISTS claim_proofs (claim_id TEXT PRIMARY KEY, proof TEXT);"""
        )
        self.db.commit()

    def _q(self, sql, args):
        return [{k.lower(): r[k] for k in r.keys()} for r in self.db.execute(sql, tuple(args))]

    def _x(self, sql, args):
        self.db.execute(sql, tuple(args))
        self.db.commit()

    @staticmethod
    def _blob(v):
        return None if v is None else np.asarray(v, dtype=np.float32).tobytes()

    @staticmethod
    def _vec(b):
        return None if b is None else np.frombuffer(b, dtype=np.float32)

    def _row(self, d) -> Post:
        d = dict(d)
        d["img_vec"], d["txt_vec"] = self._vec(d.get("img_vec")), self._vec(d.get("txt_vec"))
        return Post(**d)

    def add(self, p: Post) -> Post:
        self._x("INSERT INTO posts VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (p.id, p.kind, p.title, p.description, p.location, p.contact, p.category,
                 p.image_path, p.created_at, p.status, self._blob(p.img_vec), self._blob(p.txt_vec)))
        return p

    def matches(self, p: Post, k: int = 5) -> list[dict]:
        skip = self.rejected_pairs(p.id)
        rows = []
        for c in self.list(OPPOSITE[p.kind]):
            if c.id in skip:
                continue
            score, parts = combine(_with_sem(_sims(p, c.img_vec, c.txt_vec), p, c))
            rows.append({"post": c.public(), "score": score, "signals": parts})
        return _rank(rows, k)


class SnowflakeStore(BaseStore):
    """Requires SNOWFLAKE_ACCOUNT, SNOWFLAKE_USER, SNOWFLAKE_PASSWORD (or other auth
    supported by the connector), SNOWFLAKE_WAREHOUSE, SNOWFLAKE_DATABASE, SNOWFLAKE_SCHEMA."""

    backend = "snowflake"
    P = "%s"
    LIST_COLS = "* EXCLUDE (img_vec, txt_vec)"
    VEC = f"VECTOR(FLOAT, {DIM})"

    def __init__(self):
        import snowflake.connector

        self.conn = snowflake.connector.connect(
            account=os.environ["SNOWFLAKE_ACCOUNT"],
            user=os.environ["SNOWFLAKE_USER"],
            password=os.getenv("SNOWFLAKE_PASSWORD"),
            authenticator=os.getenv("SNOWFLAKE_AUTHENTICATOR", "snowflake"),
            role=os.getenv("SNOWFLAKE_ROLE"),
            warehouse=os.environ["SNOWFLAKE_WAREHOUSE"],
            database=os.environ["SNOWFLAKE_DATABASE"],
            schema=os.getenv("SNOWFLAKE_SCHEMA", "PUBLIC"),
        )
        self._x(f"""CREATE TABLE IF NOT EXISTS POSTS (
                ID STRING PRIMARY KEY, KIND STRING, TITLE STRING, DESCRIPTION STRING,
                LOCATION STRING, CONTACT STRING, CATEGORY STRING, IMAGE_PATH STRING,
                CREATED_AT FLOAT, STATUS STRING, IMG_VEC {self.VEC}, TXT_VEC {self.VEC})""", ())
        self._x("""CREATE TABLE IF NOT EXISTS CLAIMS (
                ID STRING PRIMARY KEY, LOST_ID STRING, FOUND_ID STRING,
                CREATED_AT FLOAT, UPDATED_AT FLOAT, STATUS STRING)""", ())
        self._x("CREATE TABLE IF NOT EXISTS POST_KEYS (POST_ID STRING PRIMARY KEY, KEY_HASH STRING)", ())
        self._x("CREATE TABLE IF NOT EXISTS CLAIM_PROOFS (CLAIM_ID STRING PRIMARY KEY, PROOF STRING)", ())

    def _cur(self):
        from snowflake.connector import DictCursor

        return self.conn.cursor(DictCursor)

    def _q(self, sql, args):
        cur = self._cur()
        cur.execute(sql, tuple(args))
        return [{k.lower(): v for k, v in r.items()} for r in cur.fetchall()]

    def _x(self, sql, args):
        self._cur().execute(sql, tuple(args))

    @staticmethod
    def _j(v):
        return None if v is None else json.dumps([round(float(x), 6) for x in v])

    def _vexpr(self):
        return f"PARSE_JSON(%s)::ARRAY::{self.VEC}"

    def _row(self, r) -> Post:
        def vec(x):
            if x is None:
                return None
            return np.asarray(json.loads(x) if isinstance(x, str) else x, dtype=np.float32)

        return Post(
            id=r["id"], kind=r["kind"], title=r["title"], description=r["description"] or "",
            location=r["location"] or "", contact=r["contact"] or "", category=r["category"] or "",
            image_path=r["image_path"] or "", created_at=r["created_at"], status=r["status"],
            img_vec=vec(r.get("img_vec")), txt_vec=vec(r.get("txt_vec")),
        )

    def add(self, p: Post) -> Post:
        self._x(f"""INSERT INTO POSTS SELECT %s,%s,%s,%s,%s,%s,%s,%s,%s,%s,
                    {self._vexpr()}, {self._vexpr()}""",
                (p.id, p.kind, p.title, p.description, p.location, p.contact, p.category,
                 p.image_path, p.created_at, p.status, self._j(p.img_vec), self._j(p.txt_vec)))
        return p

    def matches(self, p: Post, k: int = 5):
        # All similarity math happens inside Snowflake.
        skip = self.rejected_pairs(p.id)
        sql = f"""
            WITH q AS (SELECT {self._vexpr()} AS QI, {self._vexpr()} AS QT)
            SELECT P.* EXCLUDE (IMG_VEC, TXT_VEC),
                   VECTOR_COSINE_SIMILARITY(P.IMG_VEC, q.QI) AS IMG_IMG,
                   VECTOR_COSINE_SIMILARITY(P.TXT_VEC, q.QT) AS TXT_TXT,
                   VECTOR_COSINE_SIMILARITY(P.IMG_VEC, q.QT) AS CROSS_A,
                   VECTOR_COSINE_SIMILARITY(P.TXT_VEC, q.QI) AS CROSS_B
            FROM POSTS P, q
            WHERE P.KIND=%s AND P.STATUS IN ('open','claimed') AND P.CREATED_AT >= %s
            ORDER BY GREATEST(COALESCE(IMG_IMG,-1), COALESCE(TXT_TXT,-1),
                              COALESCE(CROSS_A,-1) + 0.5, COALESCE(CROSS_B,-1) + 0.5) DESC
            LIMIT 50"""
        rows = []
        for r in self._q(sql, (self._j(p.img_vec), self._j(p.txt_vec), OPPOSITE[p.kind], cutoff())):
            if r["id"] in skip:
                continue
            cross = [x for x in (r["cross_a"], r["cross_b"]) if x is not None]
            sims = {"img_img": r["img_img"], "txt_txt": r["txt_txt"], "cross": max(cross) if cross else None}
            cand = self._row(r)
            score, parts = combine(_with_sem(sims, p, cand))
            rows.append({"post": cand.public(), "score": score, "signals": parts})
        return _rank(rows, k)

    def complete(self, prompt: str, model: str | None = None) -> str:
        """Snowflake Cortex LLM call. Defaults to an open-weight model."""
        model = model or os.getenv("CORTEX_MODEL", "llama3.1-8b")
        r = self._q("SELECT SNOWFLAKE.CORTEX.COMPLETE(%s, %s) AS OUT", (model, prompt))
        return (r[0]["out"] if r else "").strip()


def get_store():
    if os.getenv("SNOWFLAKE_ACCOUNT"):
        return SnowflakeStore()
    return LocalStore(os.getenv("GRYPHFIND_DB", "gryphfind.db"))
