"""Embedding models.

ClipEmbedder uses OpenAI's open-weight CLIP (ViT-B/32) via Hugging Face transformers.
CLIP puts images and text in the same vector space, so a "lost" post that is only a
text description ("blue Hydro Flask with a Guelph sticker") can still match a
"found" post that is only a photo.

StubEmbedder is a deterministic stand-in used by the test suite so tests run
without downloading model weights.
"""
from __future__ import annotations

import hashlib
import os

import numpy as np
from PIL import Image

DIM = 512

# Zero-shot labels used to auto-tag items. Kept short so the dropdown stays usable.
CATEGORIES = [
    "water bottle", "phone", "keys", "wallet", "backpack", "laptop", "headphones",
    "earbuds case", "jacket", "hat", "umbrella", "ID card", "glasses", "jewelry",
    "book", "charger", "calculator", "watch", "shoes", "other",
]


def _normalize(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32).reshape(-1)
    n = np.linalg.norm(v)
    return v / n if n else v


class ClipEmbedder:
    name = "clip"

    def __init__(self, model_id: str | None = None):
        import torch
        from transformers import CLIPModel, CLIPProcessor

        self.torch = torch
        self.model_id = model_id or os.getenv("CLIP_MODEL", "openai/clip-vit-base-patch32")
        self.device = "cuda" if torch.cuda.is_available() else (
            "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu"
        )
        self.model = CLIPModel.from_pretrained(self.model_id).to(self.device).eval()
        self.processor = CLIPProcessor.from_pretrained(self.model_id)
        self._cat_vecs = np.stack([self.embed_text(f"a photo of a {c}") for c in CATEGORIES])

    # We call CLIP's vision/text towers and projection layers directly instead of
    # get_image_features/get_text_features, whose return type changed between
    # transformers 4.x (tensor) and 5.x (output object). The towers' pooler_output
    # plus the projection is exactly what CLIP uses internally in every version.
    def embed_image(self, img: Image.Image) -> np.ndarray:
        inputs = self.processor(images=img.convert("RGB"), return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            pooled = self.model.vision_model(pixel_values=inputs["pixel_values"]).pooler_output
            vec = self.model.visual_projection(pooled)
        return _normalize(vec[0].float().cpu().numpy())

    def embed_text(self, text: str) -> np.ndarray:
        inputs = self.processor(text=[text], return_tensors="pt", padding=True, truncation=True).to(self.device)
        with self.torch.no_grad():
            pooled = self.model.text_model(
                input_ids=inputs["input_ids"], attention_mask=inputs.get("attention_mask")
            ).pooler_output
            vec = self.model.text_projection(pooled)
        return _normalize(vec[0].float().cpu().numpy())

    def classify(self, img_vec: np.ndarray) -> str:
        sims = self._cat_vecs @ img_vec
        return CATEGORIES[int(np.argmax(sims))]


class StubEmbedder:
    """Deterministic fake embedder for tests. Similar inputs give similar vectors:
    images are embedded from a coarse colour histogram, text from word hashes."""

    name = "stub"

    def embed_image(self, img: Image.Image) -> np.ndarray:
        small = np.asarray(img.convert("RGB").resize((8, 8)), dtype=np.float32).reshape(-1)
        v = np.zeros(DIM, dtype=np.float32)
        v[: small.size] = small
        return _normalize(v + 1e-3)

    def embed_text(self, text: str) -> np.ndarray:
        v = np.zeros(DIM, dtype=np.float32)
        for w in text.lower().split():
            h = int(hashlib.md5(w.strip(".,!?").encode()).hexdigest(), 16)
            v[h % DIM] += 1.0
        return _normalize(v + 1e-3)

    def classify(self, img_vec: np.ndarray) -> str:
        return "other"


_instance = None


def get_embedder():
    global _instance
    if _instance is None:
        _instance = StubEmbedder() if os.getenv("EMBEDDER", "clip") == "stub" else ClipEmbedder()
    return _instance
