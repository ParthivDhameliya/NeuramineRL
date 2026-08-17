from __future__ import annotations

from neuraminerl import Learner
from neuraminerl.llm.fake import FakeLLM
from neuraminerl.models import Lesson
from neuraminerl.retrieval.injector import Injector


def _insert(learner: Learner, condition: str, advice: str, **kw: object) -> Lesson:
    lesson = Lesson(scope=learner.scope, condition=condition, advice=advice, **kw)  # type: ignore[arg-type]
    vector = learner._embedder.embed([lesson.text])[0]
    learner._store.upsert_lesson(lesson, embedding=vector)
    return lesson


def test_recall_renders_block(learner: Learner) -> None:
    _insert(learner, "When submitting the booking form", "Use ISO dates.")
    result = learner.recall("submit the booking form for a flight")
    assert result
    text = str(result)
    assert text.startswith("<learned_lessons>")
    assert "1. When submitting the booking form: Use ISO dates." in text
    assert result.token_count > 0


def test_recall_is_read_only(learner: Learner) -> None:
    _insert(learner, "When submitting the booking form", "Use ISO dates.")
    learner.recall("submit the booking form")
    lessons = learner.lessons()
    assert lessons[0].times_injected == 0


def test_run_recall_binds_injections(learner: Learner) -> None:
    _insert(learner, "When submitting the booking form", "Use ISO dates.")
    with learner.run(task="submit the booking form") as run:
        assert run.lessons
        run.end(success=True)
    injections = learner._store.injections_for(run.id)
    assert len(injections) == 1 and injections[0].credited
    assert learner.lessons()[0].times_injected == 1


def test_empty_recall_is_empty_string(learner: Learner) -> None:
    result = learner.recall("anything at all")
    assert str(result) == ""
    assert not result
    assert result.token_count == 0


def test_candidate_quota_reserves_slot(fake_llm: FakeLLM) -> None:
    learner = Learner(store=":memory:", embedder="hashed", llm=fake_llm, k=3)
    for _i, (condition, advice) in enumerate(
        [
            ("When booking flights on the travel site", "Check baggage fees first."),
            ("When booking hotels for the trip", "Sort results by total price."),
            ("When booking rental cars abroad", "Verify the license requirements."),
        ]
    ):
        lesson = _insert(learner, condition, advice, alpha=10.0, times_injected=5)
        lesson.state = "active"
        learner._store.upsert_lesson(lesson)
    fresh = _insert(learner, "When booking train tickets in Europe", "Reserve seats early.")
    assert fresh.state == "candidate"

    ranked = learner._retriever.retrieve("booking a trip", learner.scope)
    states = [lesson.state for lesson, _ in ranked]
    assert len(ranked) == 3
    assert states.count("candidate") >= 1  # the reserved exploration slot


def test_low_confidence_gated(learner: Learner) -> None:
    _insert(
        learner,
        "When submitting the booking form",
        "Bad advice.",
        alpha=1.0,
        beta=9.0,
        credited_trials=8.0,  # had its exploration chances, and blew them
    )
    result = learner.recall("submit the booking form")
    assert not result


def test_token_budget_drops_overflow(fake_llm: FakeLLM) -> None:
    learner = Learner(store=":memory:", embedder="hashed", llm=fake_llm, token_budget=60)
    injector = Injector(learner.config)
    lessons = [
        Lesson(scope="default", condition=f"When handling case {i}", advice="x" * 200)
        for i in range(5)
    ]
    block, included = injector.render(lessons)
    assert len(included) < 5
    assert "…" not in block  # nothing truncated mid-lesson
