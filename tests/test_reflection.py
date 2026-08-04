from __future__ import annotations

from neuraminerl import Learner
from neuraminerl.llm.fake import FakeLLM
from neuraminerl.models import Outcome, Trajectory
from neuraminerl.reflection.llm_reflector import FallbackReflector, render_transcript


def _fail(learner: Learner, task: str, error: str = "it broke") -> str:
    with learner.run(task=task) as run:
        run.log([{"role": "assistant", "content": "trying..."}])
        run.end(success=False, error=error)
    return run.id


def test_reflection_creates_lesson(learner: Learner, fake_llm: FakeLLM) -> None:
    fake_llm.queue(
        {
            "lessons": [
                {
                    "condition": "When submitting the booking form",
                    "advice": "Use ISO dates (YYYY-MM-DD).",
                    "rationale": "MM/DD/YYYY was silently rejected.",
                }
            ]
        }
    )
    _fail(learner, "book a flight", error="form rejected the date")
    lessons = learner.lessons()
    assert len(lessons) == 1
    assert lessons[0].state == "candidate"
    assert "ISO dates" in lessons[0].advice


def test_reflection_can_return_zero_lessons(learner: Learner, fake_llm: FakeLLM) -> None:
    fake_llm.queue({"lessons": []})
    _fail(learner, "flaky network")
    assert learner.lessons() == []


def test_empty_drafts_filtered(learner: Learner, fake_llm: FakeLLM) -> None:
    fake_llm.queue({"lessons": [{"condition": "", "advice": "do things", "rationale": ""}]})
    _fail(learner, "task")
    assert learner.lessons() == []


def test_duplicate_reinforces_instead_of_inserting(learner: Learner, fake_llm: FakeLLM) -> None:
    draft = {
        "condition": "When submitting the booking form",
        "advice": "Use ISO dates (YYYY-MM-DD).",
        "rationale": "r",
    }
    fake_llm.queue({"lessons": [draft]}, {"lessons": [draft]})
    first = _fail(learner, "book flight one")
    second = _fail(learner, "book flight two")
    lessons = learner.lessons()
    assert len(lessons) == 1
    assert set(lessons[0].source_trajectory_ids) == {first, second}
    # a recurrence is evidence the lesson is needed, not that it works
    assert lessons[0].alpha == 1.0 and lessons[0].beta == 1.0


def test_merge_generalizes() -> None:
    fake_llm = FakeLLM()
    # Force the merge band regardless of embedder similarity quirks.
    learner = Learner(
        store=":memory:",
        embedder="hashed",
        llm=fake_llm,
        dedup_duplicate_threshold=0.999,
        dedup_merge_threshold=0.0,
    )
    fake_llm.queue(
        {
            "lessons": [
                {
                    "condition": "When posting an order to the orders API",
                    "advice": "Include an Idempotency-Key header.",
                    "rationale": "r",
                }
            ]
        },
        # second failure: similar-but-not-identical draft
        {
            "lessons": [
                {
                    "condition": "When posting an order to the orders API endpoint",
                    "advice": "Always include the Idempotency-Key header on POST.",
                    "rationale": "r",
                }
            ]
        },
        # merge decision
        {
            "decision": "generalize",
            "condition": "When POSTing to the orders API",
            "advice": "Always send an Idempotency-Key header.",
        },
    )
    _fail(learner, "place an order")
    _fail(learner, "place another order")
    lessons = learner.lessons()
    assert len(lessons) == 1
    assert lessons[0].version == 2
    assert lessons[0].condition == "When POSTing to the orders API"


def test_fallback_reflector_without_llm() -> None:
    trajectory = Trajectory(scope="s", task="deploy the service")
    outcome = Outcome(
        trajectory_id=trajectory.id, status="failure", source="manual", detail="timeout after 30s"
    )
    drafts = FallbackReflector().reflect(trajectory, outcome)
    assert len(drafts) == 1
    assert "deploy the service" in drafts[0].condition
    assert "timeout" in drafts[0].advice


def test_transcript_truncation() -> None:
    trajectory = Trajectory(scope="s", task="t")
    from neuraminerl.models import Step

    trajectory.steps = [
        Step(trajectory_id=trajectory.id, index=i, kind="note", content="z" * 500)
        for i in range(100)
    ]
    text = render_transcript(trajectory, char_budget=2000)
    assert len(text) < 2200
    assert "truncated" in text
