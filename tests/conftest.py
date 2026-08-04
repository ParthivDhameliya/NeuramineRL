from __future__ import annotations

import pytest

from neuramine import Learner
from neuramine.llm.fake import FakeLLM


@pytest.fixture
def fake_llm() -> FakeLLM:
    return FakeLLM()


@pytest.fixture
def learner(fake_llm: FakeLLM) -> Learner:
    """In-memory learner with the dependency-free embedder and a fake LLM."""
    return Learner(store=":memory:", embedder="hashed", llm=fake_llm)
