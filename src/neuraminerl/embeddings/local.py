"""Default local embedder: model2vec static embeddings.

potion-base-8M is ~30 MB, pure-numpy at inference, <1 ms per embed — quality
is plenty for short condition/advice texts. Lazily downloaded on first use.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from ..exceptions import ConfigError
from .base import Vector, normalize

DEFAULT_MODEL = "minishlab/potion-base-8M"


class LocalEmbedder:
    def __init__(self, model_name: str = DEFAULT_MODEL) -> None:
        self.model_name = model_name
        self._model: Any = None
        self.dim = 0

    def _load(self) -> Any:
        if self._model is None:
            try:
                from model2vec import StaticModel
            except ImportError as exc:
                raise ConfigError(
                    "Local embeddings need the model2vec package. "
                    "Install it with: pip install neuraminerl[embeddings]"
                ) from exc
            self._model = StaticModel.from_pretrained(self.model_name)
            self.dim = int(self._model.dim)
        return self._model

    def embed(self, texts: Sequence[str]) -> Vector:
        model = self._load()
        vectors = model.encode(list(texts))
        return normalize(vectors.astype("float32"))
