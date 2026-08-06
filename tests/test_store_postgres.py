"""PostgresStore integration tests. Skipped unless NEURAMINERL_TEST_POSTGRES_DSN
points at a reachable Postgres (CI provides one via a service container)."""

from __future__ import annotations

import os

import numpy as np
import pytest

from neuraminerl import Learner
from neuraminerl.llm.fake import FakeLLM
from neuraminerl.models import Lesson, Outcome, Step, Trajectory, new_id

DSN = os.environ.get("NEURAMINERL_TEST_POSTGRES_DSN", "")

pytestmark = pytest.mark.skipif(not DSN, reason="NEURAMINERL_TEST_POSTGRES_DSN not set")


@pytest.fixture
def store():  # type: ignore[no-untyped-def]
    from neuraminerl.store.postgres import PostgresStore

    s = PostgresStore(DSN)
    yield s
    s.close()


def _vec(seed: int, dim: int = 8) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.normal(size=dim).astype(np.float32)
    return v / np.linalg.norm(v)


def test_trajectory_steps_outcome_roundtrip(store) -> None:  # type: ignore[no-untyped-def]
    scope = f"t-{new_id()}"
    trajectory = Trajectory(scope=scope, task="do the thing")
    store.save_trajectory(trajectory)
    store.add_steps(
        [
            Step(trajectory_id=trajectory.id, index=0, kind="note", content="hello"),
            Step(trajectory_id=trajectory.id, index=1, kind="tool_call", content="f(1)"),
        ]
    )
    loaded = store.get_trajectory(trajectory.id)
    assert loaded.task == "do the thing"
    assert [s.content for s in loaded.steps] == ["hello", "f(1)"]

    assert store.latest_outcome(trajectory.id) is None
    store.save_outcome(Outcome(trajectory_id=trajectory.id, status="failure", source="manual"))
    store.save_outcome(Outcome(trajectory_id=trajectory.id, status="success", source="manual"))
    latest = store.latest_outcome(trajectory.id)
    assert latest is not None
    assert latest.status == "success"


def test_lesson_upsert_search_and_injections(store) -> None:  # type: ignore[no-untyped-def]
    scope = f"t-{new_id()}"
    trajectory = Trajectory(scope=scope, task="task")
    store.save_trajectory(trajectory)

    lessons = [Lesson(scope=scope, condition=f"When c{i}", advice=f"Do a{i}.") for i in range(3)]
    for i, lesson in enumerate(lessons):
        store.upsert_lesson(lesson, embedding=_vec(i))
    assert store.count_lessons(scope, ("candidate",)) == 3

    query = _vec(1)
    results = store.search_lessons(query, scope, ("candidate", "active"), k=2)
    assert len(results) == 2
    assert results[0][0].id == lessons[1].id  # exact vector match ranks first
    assert results[0][1] == pytest.approx(1.0, abs=1e-5)

    # Update without embedding keeps the stored vector.
    lessons[1].state = "active"
    store.upsert_lesson(lessons[1])
    assert store.get_lesson(lessons[1].id).state == "active"

    from neuraminerl.models import Injection

    injection = Injection(
        trajectory_id=trajectory.id,
        lesson_id=lessons[1].id,
        lesson_version=1,
        retrieval_score=0.9,
        rank=0,
    )
    store.log_injections([injection])
    loaded = store.injections_for(trajectory.id)
    assert len(loaded) == 1
    assert not loaded[0].credited
    injection.credited = True
    injection.credited_alpha = 1.0
    store.update_injection(injection)
    assert store.injections_for(trajectory.id)[0].credited


def test_baseline_success_rate(store) -> None:  # type: ignore[no-untyped-def]
    scope = f"t-{new_id()}"
    assert store.baseline_success_rate(scope) == 0.5  # < 5 outcomes
    for i in range(6):
        trajectory = Trajectory(scope=scope, task=f"t{i}")
        store.save_trajectory(trajectory)
        status = "success" if i % 2 == 0 else "failure"
        store.save_outcome(Outcome(trajectory_id=trajectory.id, status=status, source="manual"))
    assert store.baseline_success_rate(scope) == pytest.approx(0.5)


def test_learner_end_to_end_on_postgres() -> None:
    scope = f"t-{new_id()}"
    reflection = {
        "lessons": [{"condition": "When testing", "advice": "Do it right.", "rationale": "broke"}]
    }
    nm = Learner(store=DSN, embedder="hashed", llm=FakeLLM(responses=[reflection]), scope=scope)
    with nm.run(task="test the thing") as run:
        run.log("attempt")
        run.end(success=False, error="boom")
    assert nm.lessons()

    with nm.run(task="test the thing") as run:
        assert run.lessons
        run.end(success=True)
    assert nm.stats().injected_tokens_estimate > 0
    nm.close()
