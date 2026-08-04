from __future__ import annotations

import pytest

from neuramine import Learner
from neuramine.llm.fake import FakeLLM


def test_success_run(learner: Learner) -> None:
    with learner.run(task="say hello") as run:
        run.log([{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}])
        run.end(success=True)
    trajectory = learner._store.get_trajectory(run.id)
    assert trajectory.status == "completed"
    outcome = learner._store.latest_outcome(run.id)
    assert outcome is not None and outcome.status == "success"
    assert [s.kind for s in trajectory.steps] == ["user_message", "agent_message"]


def test_unhandled_exception_becomes_failure(learner: Learner, fake_llm: FakeLLM) -> None:
    with pytest.raises(ValueError, match="kaboom"), learner.run(task="explode") as run:
        raise ValueError("kaboom")
    outcome = learner._store.latest_outcome(run.id)
    assert outcome is not None
    assert outcome.status == "failure"
    assert outcome.source == "exception"
    assert "kaboom" in outcome.detail
    # failure triggered one reflection call
    assert len(fake_llm.calls) == 1


def test_exit_without_end_is_abandoned(learner: Learner) -> None:
    with learner.run(task="wander off") as run:
        run.note("did some stuff")
    trajectory = learner._store.get_trajectory(run.id)
    assert trajectory.status == "abandoned"
    assert learner._store.latest_outcome(run.id) is None


def test_log_accepts_strings_and_blocks(learner: Learner) -> None:
    with learner.run(task="shapes") as run:
        run.log("plain note")
        run.log({"role": "assistant", "content": [{"type": "text", "text": "block text"}]})
        run.log_tool_call("search", {"q": "x"}, result={"hits": 3}, error=None)
        run.end(success=True)
    steps = learner._store.get_trajectory(run.id).steps
    kinds = [s.kind for s in steps]
    assert kinds == ["note", "agent_message", "tool_call", "tool_result"]
    assert "block text" in steps[1].content
    assert "search" in steps[2].content


def test_step_content_capped(learner: Learner) -> None:
    with learner.run(task="big") as run:
        run.log("x" * 100_000)
        run.end(success=True)
    steps = learner._store.get_trajectory(run.id).steps
    assert len(steps[0].content) == learner.config.max_step_chars


def test_partial_outcome_from_score(learner: Learner) -> None:
    with learner.run(task="partial") as run:
        run.end(score=0.7)
    outcome = learner._store.latest_outcome(run.id)
    assert outcome is not None
    assert outcome.status == "partial"
    assert outcome.score == 0.7


def test_end_idempotent(learner: Learner) -> None:
    with learner.run(task="twice") as run:
        run.end(success=True)
        run.end(success=False)  # ignored
    outcome = learner._store.latest_outcome(run.id)
    assert outcome is not None and outcome.status == "success"
