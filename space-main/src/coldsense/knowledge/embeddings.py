"""Embedding providers. The provider is replaceable: ingestion stores the model id with every vector, and
retrieval refuses to compare vectors from different models.

* ``lexical-hash-v1``, a LEXICAL embedding (feature hashing of stemmed word unigrams and bigrams, sublinear term
  frequency, L2-normalised). Real and reproducible, but it matches words, not meaning: a query that uses
  different words from the manual ("compressor will not start" vs "unit fails to run") is not matched.
* ``bge-small-en-v1.5`` (``semantic.py``), a SEMANTIC embedding run locally from pinned, hash-checked ONNX assets;
  it needs the optional ``onnxruntime`` and ``tokenizers`` packages and the model files.

A provider may define ``embed_query`` (a query-side policy such as an instruction prefix); passages always go
through ``embed``."""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from typing import Literal, Protocol

import numpy as np

from coldsense.core.errors import ColdSenseError

SEMANTIC_BLOCKER = ("the semantic provider bge-small-en-v1.5 needs its pinned model files (see "
                    "docs/KNOWLEDGE.md §1) and the optional packages onnxruntime and tokenizers")


class EmbeddingError(ColdSenseError):
    """The embedding backend is unavailable or failed; nothing is indexed or answered with a partial result."""


class EmbeddingProvider(Protocol):
    model_id: str
    kind: Literal["LEXICAL", "SEMANTIC"]
    dim: int

    def embed(self, texts: list[str]) -> np.ndarray:
        """float32 array (len(texts), dim), rows L2-normalised (zero rows for texts without terms)."""
        ...


# ---------------------------------------------------------------- shared lexical analysis
STOPWORDS = frozenset("""a an and are as at be been but by can could did do does doing for from had has have how i if
in into is it its may might must no not of on or our shall should so than that the their them then there these they
this those to was we were what when where which while who why will with would you your yes about after before above
below over under up down out off again further once here all any both each few more most other some such only own same
too very just also per via""".split())

_TOKEN = re.compile(r"\d+(?:[.,]\d+)?|[^\W\d_]+", re.UNICODE)


def stem(w: str) -> str:
    """A light, deterministic English suffix stripper (plurals, -ing, -ed). Numbers are kept as written."""
    if w[0].isdigit() or len(w) <= 3:
        return w
    if w.endswith("ies") and len(w) > 4:
        w = w[:-3] + "y"
    elif w.endswith(("sses", "shes", "ches", "xes", "zes")):
        w = w[:-2]
    elif w.endswith("s") and not w.endswith(("ss", "us", "is")):
        w = w[:-1]
    base = w
    if w.endswith("ing") and len(w) >= 6:
        w = w[:-3]
    elif w.endswith("ed") and len(w) >= 5 and not w.endswith("eed"):
        w = w[:-2]
    if w != base and len(w) >= 3 and w[-1] == w[-2] and w[-1] not in "aeiouslz":
        w = w[:-1]                                       # resetting -> resett -> reset; running -> run
    return w


def terms(text: str) -> list[str]:
    """Content terms: lower-cased, stop words removed, lightly stemmed."""
    return [stem(t) for t in (m.group(0).lower() for m in _TOKEN.finditer(text)) if t not in STOPWORDS]


def _h(feature: str) -> int:
    return int.from_bytes(hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest(), "big")


class LexicalHashingProvider:
    kind: Literal["LEXICAL", "SEMANTIC"] = "LEXICAL"

    def __init__(self, dim: int = 4096) -> None:
        if dim < 256 or dim & (dim - 1):
            raise EmbeddingError("dim must be a power of two, at least 256")
        self.dim = dim
        self.model_id = f"lexical-hash-v1/{dim}"

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            t = terms(text)
            feats = Counter(t) + Counter(f"{a} {b}" for a, b in zip(t, t[1:], strict=False))
            for f, n in feats.items():
                h = _h(f)
                out[i, h % self.dim] += (1.0 if (h >> 63) & 1 else -1.0) * (1.0 + math.log(n))
            norm = float(np.linalg.norm(out[i]))
            if norm > 0:
                out[i] /= norm
        return out


PROVIDERS = {"lexical-hash-v1": LexicalHashingProvider}


def get_provider(name: str = "lexical-hash-v1") -> EmbeddingProvider:
    if name.startswith("bge-small-en-v1.5"):
        from .semantic import from_model_id
        return from_model_id(name)
    base = name.split("/", 1)[0]
    if base not in PROVIDERS:
        raise EmbeddingError(f"embedding provider '{name}' is not available here; available: "
                             f"{', '.join(PROVIDERS)}, bge-small-en-v1.5")
    dim = int(name.split("/", 1)[1]) if "/" in name else 4096
    return PROVIDERS[base](dim)


def check_vectors(provider: EmbeddingProvider, texts: list[str], query: bool = False) -> np.ndarray:
    """Embed and verify the shape and values; any provider failure becomes an EmbeddingError. ``query`` uses the
    provider's query policy (``embed_query``) when it has one."""
    fn = getattr(provider, "embed_query", None) if query else None
    try:
        v = np.asarray((fn or provider.embed)(texts), dtype=np.float32)
    except EmbeddingError:
        raise
    except Exception as e:
        raise EmbeddingError(f"embedding provider {provider.model_id} failed ({type(e).__name__})") from e
    if v.shape != (len(texts), provider.dim) or not np.all(np.isfinite(v)):
        raise EmbeddingError(f"embedding provider {provider.model_id} returned an invalid array {v.shape}")
    return v
