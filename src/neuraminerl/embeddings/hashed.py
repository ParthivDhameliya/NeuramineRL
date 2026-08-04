"""Dependency-free fallback embedder using feature hashing.

Deterministic and offline — backs the test suite and air-gapped installs.
Retrieval quality is crude but adequate for keyword-ish task descriptions.
"""

from __future__ import annotations

import hashlib
import itertools
import re
from collections.abc import Sequence

import numpy as np

from .base import Vector, normalize

_TOKEN_RE = re.compile(r"[a-z0-9]+")


class HashedEmbedder:
    def __init__(self, dim: int = 256) -> None:
        self.dim = dim

    def embed(self, texts: Sequence[str]) -> Vector:
        matrix = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            tokens = _TOKEN_RE.findall(text.lower())
            # Unigrams + bigrams so word order carries a little signal.
            grams = tokens + [f"{a}_{b}" for a, b in itertools.pairwise(tokens)]
            for gram in grams:
                digest = hashlib.blake2b(gram.encode(), digest_size=8).digest()
                bucket = int.from_bytes(digest[:4], "little") % self.dim
                sign = 1.0 if digest[4] % 2 == 0 else -1.0
                matrix[row, bucket] += sign
        return normalize(matrix)
