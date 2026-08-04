from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt

Vector = npt.NDArray[np.float32]


@runtime_checkable
class Embedder(Protocol):
    """Embeds short texts into L2-normalized float32 vectors of shape (n, dim)."""

    dim: int

    def embed(self, texts: Sequence[str]) -> Vector: ...


def normalize(matrix: Vector) -> Vector:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return (matrix / norms).astype(np.float32)
