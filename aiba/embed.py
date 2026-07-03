"""
Optional embedding infrastructure, the [embed] extra (PLANNING.md §2.1).

Policy, enforced here so detectors don't each reinvent it:
- Model: BAAI/bge-small-en-v1.5, REVISION PINNED below. ~130MB one-time HF
  download, then fully offline; sub-5ms per embedding on a modern CPU.
- The committed data/sample/embeddings.npz gives cloners bit-exact eval
  reproduction without downloading the model. Regeneration tolerates small
  float variance; thresholds are set with margin, never at the third
  decimal.
- Embed DESCRIPTIONS and goal/planner text, never tool names. Names are
  arbitrary identifiers (fetch_hr_recs); descriptions carry the semantics.
- Budget: under 10 embeddings per session (goal, planner summaries, tool
  justifications, final answer).

Import of sentence-transformers happens lazily inside functions so the
rule-based core imports this module's cache/IO helpers without the extra.
"""

from __future__ import annotations

import hashlib
import math
import re

MODEL_NAME = "BAAI/bge-small-en-v1.5"
MODEL_REVISION = "PIN-ME"  # set to exact HF commit hash when wiring the real model

_FALLBACK_DIM = 256
_STOP = {"the", "a", "an", "and", "or", "for", "to", "of", "in", "on", "this",
         "our", "with", "from", "by", "into", "as", "at", "list", "run"}
_model = None  # lazily-loaded SentenceTransformer, if the [embed] extra is present


def _load_model():
    """Import + load the pinned model on first use. Returns None if the
    [embed] extra isn't installed (the rule-based core never calls this)."""
    global _model
    if _model is None:
        try:
            from sentence_transformers import SentenceTransformer
            _model = SentenceTransformer(MODEL_NAME, revision=None if MODEL_REVISION == "PIN-ME"
                                         else MODEL_REVISION)
        except Exception:
            _model = False  # sentinel: extra unavailable, use deterministic fallback
    return _model or None


def using_real_model() -> bool:
    return _load_model() is not None


def _fallback_embed(text: str) -> list[float]:
    """Deterministic hashed bag-of-words embedding (no model download).

    Fully offline and seed-free-deterministic, so goal_drift / rug-pull run
    everywhere and the committed sample eval is reproducible. Cosine here
    tracks vocabulary overlap, which is enough to separate on-topic planner steps from
    a drifted/exfiltration plan on the AUTHORED sample text. The real bge model
    replaces this transparently when `pip install .[embed]` is present."""
    vec = [0.0] * _FALLBACK_DIM
    for tok in re.findall(r"[a-z0-9]+", text.lower()):
        if tok in _STOP or len(tok) < 2:
            continue
        idx = int(hashlib.sha1(tok.encode()).hexdigest(), 16) % _FALLBACK_DIM
        vec[idx] += 1.0
    norm = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / norm for x in vec]


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed texts with the pinned model, or the deterministic fallback."""
    m = _load_model()
    if m is not None:
        return [list(v) for v in m.encode(texts, normalize_embeddings=True)]
    return [_fallback_embed(t) for t in texts]


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(y * y for y in b)) or 1.0
    return dot / (na * nb)


def save_npz(path: str, keys: list[str], vectors: list[list[float]]) -> None:
    """Write the committed embeddings cache (data/sample/embeddings.npz)."""
    import numpy as np
    np.savez(path, keys=np.array(keys), vectors=np.array(vectors))


def load_npz(path: str) -> dict[str, list[float]]:
    """Load precomputed embeddings; detectors prefer this over the model."""
    import numpy as np
    d = np.load(path, allow_pickle=True)
    return {k: list(v) for k, v in zip(d["keys"], d["vectors"])}


def embeddings_available() -> bool:
    """Always True: either the [embed] model or the deterministic fallback.
    Use using_real_model() to tell which path is active."""
    return True
