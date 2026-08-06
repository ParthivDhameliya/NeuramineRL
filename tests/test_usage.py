from __future__ import annotations

import pytest

from neuraminerl import Learner, UsageEvent
from neuraminerl.llm.base import LLMResponse
from neuraminerl.llm.fake import FakeLLM

REFLECTION = {
    "lessons": [
        {"condition": "When testing", "advice": "Do the thing right.", "rationale": "it broke"}
    ]
}


def test_on_usage_fires_on_reflection() -> None:
    events: list[UsageEvent] = []
    fake = FakeLLM(
        responses=[LLMResponse(data=REFLECTION, usage={"input_tokens": 100, "output_tokens": 20})]
    )
    nm = Learner(store=":memory:", embedder="hashed", llm=fake, on_usage=events.append)
    with nm.run(task="test the thing") as run:
        run.end(success=False, error="boom")
    assert len(events) == 1
    event = events[0]
    assert event.client == "FakeLLM"
    assert event.model == "fake"
    assert event.input_tokens == 100
    assert event.output_tokens == 20


def test_on_usage_errors_never_break_the_run() -> None:
    def explode(event: UsageEvent) -> None:
        raise RuntimeError("billing backend down")

    fake = FakeLLM(responses=[REFLECTION])
    nm = Learner(store=":memory:", embedder="hashed", llm=fake, on_usage=explode)
    with pytest.warns(UserWarning, match="on_usage callback raised"), nm.run(task="t") as run:
        run.end(success=False, error="boom")
    # Reflection still happened despite the raising callback.
    assert nm.lessons()
