from __future__ import annotations

import numpy as np

from neuramine.embeddings.hashed import HashedEmbedder


def test_deterministic() -> None:
    emb = HashedEmbedder()
    a = emb.embed(["submit the booking form"])
    b = emb.embed(["submit the booking form"])
    assert np.allclose(a, b)


def test_normalized() -> None:
    emb = HashedEmbedder()
    vectors = emb.embed(["one", "two words here", ""])
    norms = np.linalg.norm(vectors, axis=1)
    # empty text embeds to the zero vector; others are unit length
    assert np.allclose(norms[:2], 1.0, atol=1e-5)
    assert norms.shape == (3,)


def test_similar_texts_score_higher() -> None:
    emb = HashedEmbedder()
    vectors = emb.embed(
        [
            "book a flight from NYC to SFO",
            "book a cheap flight NYC to SFO tomorrow",
            "bake a chocolate cake",
        ]
    )
    sim_related = float(vectors[0] @ vectors[1])
    sim_unrelated = float(vectors[0] @ vectors[2])
    assert sim_related > sim_unrelated
