"""Turn raw CLIP cosine similarities into one 0-100 match confidence.

CLIP similarities live on different scales depending on what is compared:
image-to-image scores for the same object are typically 0.80-0.95, text-to-text
0.80-0.95, but image-to-text scores top out around 0.30-0.35. So each pair type
gets its own logistic calibration before we combine them.
Tune these with real photos at the event: `python -m gryphfind.calibrate`.
"""
from __future__ import annotations

import math

# (midpoint, temperature) per comparison type
CALIBRATION = {
    "img_img": (0.78, 0.04),
    "txt_txt": (0.82, 0.04),
    "cross": (0.255, 0.015),
}
WEIGHTS = {"img_img": 0.6, "txt_txt": 0.25, "cross": 0.15}


def _sig(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def calibrate(kind: str, sim: float | None) -> float | None:
    if sim is None:
        return None
    mid, temp = CALIBRATION[kind]
    return _sig((sim - mid) / temp)


def combine(sims: dict[str, float | None]) -> tuple[int, dict[str, int]]:
    """sims: {"img_img": s, "txt_txt": s, "cross": s} with None where unavailable.
    Returns (overall score 0-100, per-signal breakdown 0-100)."""
    parts = {k: calibrate(k, v) for k, v in sims.items() if v is not None}
    if not parts:
        return 0, {}
    total_w = sum(WEIGHTS[k] for k in parts)
    weighted = sum(WEIGHTS[k] * p for k, p in parts.items()) / total_w
    # A single very strong signal (e.g. near-identical photos) should not be
    # dragged down by a vague text description.
    score = max(weighted, 0.9 * max(parts.values()))
    return round(score * 100), {k: round(p * 100) for k, p in parts.items()}
