"""The whole loop, end to end, with a fake LLM and hashed embeddings:
fail -> reflect -> lesson -> inject -> succeed -> promote."""

from __future__ import annotations

from neuramine import Learner
from neuramine.llm.fake import FakeLLM


def test_full_loop() -> None:
    fake_llm = FakeLLM()
    learner = Learner(store=":memory:", embedder="hashed", llm=fake_llm)

    # Episode 1: the agent fails; neuramine reflects.
    fake_llm.queue(
        {
            "lessons": [
                {
                    "condition": "When placing an order via the orders API",
                    "advice": "Include an Idempotency-Key header on every POST.",
                    "rationale": "The API returned 400 without it.",
                }
            ]
        }
    )
    with learner.run(task="place an order for 3 widgets") as run:
        assert not run.lessons  # nothing learned yet
        run.log([{"role": "assistant", "content": "POST /orders -> 400 missing header"}])
        run.end(success=False, error="400 missing Idempotency-Key header")

    lessons = learner.lessons()
    assert len(lessons) == 1 and lessons[0].state == "candidate"

    # Episodes 2-4: the lesson is injected; the agent now succeeds.
    for i in range(3):
        with learner.run(task=f"place an order for {i} gadgets") as run:
            block = str(run.lessons)
            assert "Idempotency-Key" in block
            run.end(success=True)

    lesson = learner.lessons()[0]
    assert lesson.state == "active"  # promoted: injected 3x, beat the baseline
    assert lesson.times_injected == 3
    assert lesson.alpha == 4.0

    stats = learner.stats()
    assert stats.lessons_by_state["active"] == 1


def test_degraded_mode_without_llm(monkeypatch: object) -> None:
    """No LLM key at all: still works, stores raw failure observations."""
    import pytest

    for var in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "NEURAMINE_LLM"):
        monkeypatch.delenv(var, raising=False)  # type: ignore[attr-defined]

    with pytest.warns(UserWarning, match="No reflection LLM"):
        learner = Learner(store=":memory:", embedder="hashed")
    with learner.run(task="deploy the payment service") as run:
        run.end(success=False, error="TimeoutError: connect timed out after 30s")
    lessons = learner.lessons()
    assert len(lessons) == 1
    assert "deploy the payment service" in lessons[0].condition
