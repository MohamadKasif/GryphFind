"""Storage + vector matching.

LocalStore   - SQLite file, similarity computed in numpy. Zero setup.
SnowflakeStore - posts live in a Snowflake table with VECTOR(FLOAT, 512) columns and
                 similarity is computed in the warehouse with VECTOR_COSINE_SIMILARITY.

Both expose the same interface so app.py doesn't care which one is active.
"""
from __future__ import annotations

import json
import os
import sqlite3
import time
import uuid
from dataclasses import asdict, dataclass, field

import numpy as np

from .embedder import DIM
from .scoring import combine

OPPOSITE = {"lost": "found", "found": "lost"}


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
    status: str = "open"
    img_vec: np.ndarray | None = None
    txt_vec: np.ndarray | None = None

    def public(self) -> dict:
        d = asdict(self)
        d.pop("img_vec"), d.pop("txt_vec")
        d["image_url"] = f"/uploads/{self.image_path}" if self.image_path else None
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


def _rank(rows, k):
    rows.sort(key=lambda r: r["score"], reverse=True)
    return rows[:k]


class LocalStore:
    backend = "sqlite"

    def __init__(self, path: str = "gryphfind.db"):
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.execute(
            """CREATE TABLE IF NOT EXISTS posts (
                id TEXT PRIMARY KEY, kind TEXT, title TEXT, description TEXT,
                location TEXT, contact TEXT, category TEXT, image_path TEXT,
                created_at REAL, status TEXT, img_vec BLOB, txt_vec BLOB)"""
        )
        self.db.commit()

    @staticmethod
    def _blob(v):
        return None if v is None else np.asarray(v, dtype=np.float32).tobytes()

    @staticmethod
    def _vec(b):
        return None if b is None else np.frombuffer(b, dtype=np.float32)

    def _row(self, r) -> Post:
        d = dict(r)
        d["img_vec"], d["txt_vec"] = self._vec(d["img_vec"]), self._vec(d["txt_vec"])
        return Post(**d)

    def add(self, p: Post) -> Post:
        self.db.execute(
            "INSERT INTO posts VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (p.id, p.kind, p.title, p.description, p.location, p.contact, p.category,
             p.image_path, p.created_at, p.status, self._blob(p.img_vec), self._blob(p.txt_vec)),
        )
        self.db.commit()
        return p

    def get(self, pid: str) -> Post | None:
        r = self.db.execute("SELECT * FROM posts WHERE id=?", (pid,)).fetchone()
        return self._row(r) if r else None

    def list(self, kind: str | None = None, status: str = "open") -> list[Post]:
        q, args = "SELECT * FROM posts WHERE status=?", [status]
        if kind:
            q += " AND kind=?"
            args.append(kind)
        return [self._row(r) for r in self.db.execute(q + " ORDER BY created_at DESC", args)]

    def resolve(self, pid: str) -> None:
        self.db.execute("UPDATE posts SET status='resolved' WHERE id=?", (pid,))
        self.db.commit()

    def matches(self, p: Post, k: int = 5) -> list[dict]:
        rows = []
        for c in self.list(OPPOSITE[p.kind]):
            score, parts = combine(_sims(p, c.img_vec, c.txt_vec))
            rows.append({"post": c.public(), "score": score, "signals": parts})
        return _rank(rows, k)

    def stats(self) -> dict:
        cur = self.db.execute
        return {
            "open_lost": cur("SELECT COUNT(*) FROM posts WHERE kind='lost' AND status='open'").fetchone()[0],
            "open_found": cur("SELECT COUNT(*) FROM posts WHERE kind='found' AND status='open'").fetchone()[0],
            "reunited": cur("SELECT COUNT(*) FROM posts WHERE status='resolved' AND kind='lost'").fetchone()[0],
            "by_category": [dict(r) for r in cur(
                "SELECT category, COUNT(*) AS n FROM posts GROUP BY category ORDER BY n DESC LIMIT 8")],
            "by_location": [dict(r) for r in cur(
                "SELECT location, COUNT(*) AS n FROM posts WHERE location<>'' GROUP BY location ORDER BY n DESC LIMIT 8")],
        }


class SnowflakeStore:
    """Requires SNOWFLAKE_ACCOUNT, SNOWFLAKE_USER, SNOWFLAKE_PASSWORD (or other auth
    supported by the connector), SNOWFLAKE_WAREHOUSE, SNOWFLAKE_DATABASE, SNOWFLAKE_SCHEMA."""

    backend = "snowflake"
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
        self._exec(
            f"""CREATE TABLE IF NOT EXISTS POSTS (
                ID STRING PRIMARY KEY, KIND STRING, TITLE STRING, DESCRIPTION STRING,
                LOCATION STRING, CONTACT STRING, CATEGORY STRING, IMAGE_PATH STRING,
                CREATED_AT FLOAT, STATUS STRING, IMG_VEC {self.VEC}, TXT_VEC {self.VEC})"""
        )

    def _exec(self, sql, args=None):
        cur = self.conn.cursor(snowflake_dict_cursor())
        cur.execute(sql, args or ())
        return cur

    @staticmethod
    def _j(v):
        return None if v is None else json.dumps([round(float(x), 6) for x in v])

    def _vexpr(self, placeholder="%s"):
        return f"PARSE_JSON({placeholder})::ARRAY::{self.VEC}"

    def _row(self, r) -> Post:
        def vec(x):
            if x is None:
                return None
            return np.asarray(json.loads(x) if isinstance(x, str) else x, dtype=np.float32)

        return Post(
            id=r["ID"], kind=r["KIND"], title=r["TITLE"], description=r["DESCRIPTION"] or "",
            location=r["LOCATION"] or "", contact=r["CONTACT"] or "", category=r["CATEGORY"] or "",
            image_path=r["IMAGE_PATH"] or "", created_at=r["CREATED_AT"], status=r["STATUS"],
            img_vec=vec(r.get("IMG_VEC")), txt_vec=vec(r.get("TXT_VEC")),
        )

    def add(self, p: Post) -> Post:
        self._exec(
            f"""INSERT INTO POSTS SELECT %s,%s,%s,%s,%s,%s,%s,%s,%s,%s,
                {self._vexpr()}, {self._vexpr()}""",
            (p.id, p.kind, p.title, p.description, p.location, p.contact, p.category,
             p.image_path, p.created_at, p.status, self._j(p.img_vec), self._j(p.txt_vec)),
        )
        return p

    def get(self, pid):
        r = self._exec("SELECT * FROM POSTS WHERE ID=%s", (pid,)).fetchone()
        return self._row(r) if r else None

    def list(self, kind=None, status="open"):
        q, args = "SELECT * EXCLUDE (IMG_VEC, TXT_VEC) FROM POSTS WHERE STATUS=%s", [status]
        if kind:
            q += " AND KIND=%s"
            args.append(kind)
        return [self._row(r) for r in self._exec(q + " ORDER BY CREATED_AT DESC", args)]

    def resolve(self, pid):
        self._exec("UPDATE POSTS SET STATUS='resolved' WHERE ID=%s", (pid,))

    def matches(self, p: Post, k: int = 5):
        # All similarity math happens inside Snowflake.
        qi, qt = self._j(p.img_vec), self._j(p.txt_vec)
        sql = f"""
            WITH q AS (SELECT {self._vexpr()} AS QI, {self._vexpr()} AS QT)
            SELECT P.* EXCLUDE (IMG_VEC, TXT_VEC),
                   VECTOR_COSINE_SIMILARITY(P.IMG_VEC, q.QI) AS IMG_IMG,
                   VECTOR_COSINE_SIMILARITY(P.TXT_VEC, q.QT) AS TXT_TXT,
                   VECTOR_COSINE_SIMILARITY(P.IMG_VEC, q.QT) AS CROSS_A,
                   VECTOR_COSINE_SIMILARITY(P.TXT_VEC, q.QI) AS CROSS_B
            FROM POSTS P, q
            WHERE P.KIND=%s AND P.STATUS='open'
            ORDER BY GREATEST(COALESCE(IMG_IMG,-1), COALESCE(TXT_TXT,-1),
                              COALESCE(CROSS_A,-1) + 0.5, COALESCE(CROSS_B,-1) + 0.5) DESC
            LIMIT 50"""
        rows = []
        for r in self._exec(sql, (qi, qt, OPPOSITE[p.kind])):
            cross = [x for x in (r["CROSS_A"], r["CROSS_B"]) if x is not None]
            sims = {"img_img": r["IMG_IMG"], "txt_txt": r["TXT_TXT"], "cross": max(cross) if cross else None}
            score, parts = combine(sims)
            rows.append({"post": self._row(r).public(), "score": score, "signals": parts})
        return _rank(rows, k)

    def stats(self):
        one = lambda sql: list(self._exec(sql).fetchone().values())[0]
        return {
            "open_lost": one("SELECT COUNT(*) FROM POSTS WHERE KIND='lost' AND STATUS='open'"),
            "open_found": one("SELECT COUNT(*) FROM POSTS WHERE KIND='found' AND STATUS='open'"),
            "reunited": one("SELECT COUNT(*) FROM POSTS WHERE STATUS='resolved' AND KIND='lost'"),
            "by_category": [{"category": r["CATEGORY"], "n": r["N"]} for r in self._exec(
                "SELECT CATEGORY, COUNT(*) AS N FROM POSTS GROUP BY CATEGORY ORDER BY N DESC LIMIT 8")],
            "by_location": [{"location": r["LOCATION"], "n": r["N"]} for r in self._exec(
                "SELECT LOCATION, COUNT(*) AS N FROM POSTS WHERE LOCATION<>'' GROUP BY LOCATION ORDER BY N DESC LIMIT 8")],
        }

    def complete(self, prompt: str, model: str | None = None) -> str:
        """Snowflake Cortex LLM call. Defaults to an open-weight model."""
        model = model or os.getenv("CORTEX_MODEL", "llama3.1-8b")
        r = self._exec("SELECT SNOWFLAKE.CORTEX.COMPLETE(%s, %s) AS OUT", (model, prompt)).fetchone()
        return (r["OUT"] or "").strip()


def snowflake_dict_cursor():
    from snowflake.connector import DictCursor

    return DictCursor


def get_store():
    if os.getenv("SNOWFLAKE_ACCOUNT"):
        return SnowflakeStore()
    return LocalStore(os.getenv("GRYPHFIND_DB", "gryphfind.db"))
