"""One-sentence "why we think this is yours" blurbs for top matches.

Order of preference:
1. Snowflake Cortex COMPLETE with an open-weight model (when Snowflake is configured)
2. A local open-weight model through Ollama (set OLLAMA_HOST, e.g. http://localhost:11434)
3. A plain template built from the match signals (always works, no LLM)
"""
from __future__ import annotations

import os

import requests

PROMPT = """You help reunite people with lost items on a university campus.
A {qkind} post and a {ckind} post were matched by an image-similarity model.
{qkind} post: title="{qt}", details="{qd}", location="{ql}", category="{qc}"
{ckind} post: title="{ct}", details="{cd}", location="{cl}", category="{cc}"
Match confidence: {score}/100.
In ONE short friendly sentence (max 25 words), say what lines up between them and
one thing the owner should verify in person. No preamble."""


def _template(q, c, score, signals) -> str:
    bits = []
    if signals.get("img_img", 0) >= 60:
        bits.append("the photos look alike")
    if signals.get("cross", 0) >= 60:
        bits.append("the photo fits the description")
    if signals.get("txt_txt", 0) >= 60:
        bits.append("the descriptions agree")
    if q.category and q.category == c.category:
        bits.append(f"both are tagged {q.category}")
    if q.location and c.location and q.location.lower() == c.location.lower():
        bits.append(f"both were near {c.location}")
    if not bits:
        return "Weak match. Worth a look, but check a unique detail before claiming."
    return "Possible match: " + ", ".join(bits) + "."


def explain(store, q, c, score: int, signals: dict) -> str:
    prompt = PROMPT.format(
        qkind=q.kind, ckind=c.kind, qt=q.title, qd=q.description, ql=q.location, qc=q.category,
        ct=c.title, cd=c.description, cl=c.location, cc=c.category, score=score,
    )
    try:
        if hasattr(store, "complete"):
            out = store.complete(prompt)
            if out:
                return out
        host = os.getenv("OLLAMA_HOST")
        if host:
            r = requests.post(
                f"{host.rstrip('/')}/api/generate",
                json={"model": os.getenv("OLLAMA_MODEL", "llama3.2:3b"), "prompt": prompt, "stream": False},
                timeout=20,
            )
            r.raise_for_status()
            out = r.json().get("response", "").strip()
            if out:
                return out
    except Exception as e:  # never let the LLM break matching
        print(f"[explain] LLM unavailable, using template: {e}")
    return _template(q, c, score, signals)
