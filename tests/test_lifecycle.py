from __future__ import annotations

import pytest

from neuraminerl import Learner
from neuraminerl.lessons.scoring import beta_lower_bound, decay
from neuraminerl.llm.fake import FakeLLM
from neuraminerl.models import Outcome, Trajectory


def _seed_lesson(learner: Learner, fake_llm: FakeLLM, condition: str, advice: str) -> str:
    """Create one lesson by failing a run."""
    fake_llm.queue({"lessons": [{"condition": condition, "advice": advice, "rationale": "r"}]})
    with learner.run(task=f"seed {condition}") as run:
        run.end(success=False, error="failed")
    matches = [lesson for lesson in learner.lessons() if lesson.condition == condition]
    assert matches, f"seeding failed for: {condition}"
    return matches[0].id


def _run_with_lesson(learner: Learner, task: str, *, success: bool) -> None:
    with learner.run(task=task) as run:
        _ = run.lessons  # binds injections
        run.end(success=success)


def _seed_baseline(learner: Learner, n_success: int, n_failure: int = 0) -> None:
    """Outcomes with no injections, to move the scope baseline."""
    store = learner._store
    for i in range(n_success + n_failure):
        t = Trajectory(scope=learner.scope, task=f"baseline {i}")
        store.save_trajectory(t)
        store.save_outcome(
            Outcome(
                trajectory_id=t.id,
                status="success" if i < n_success else "failure",
                source="manual",
            )
        )


def test_scoring_math() -> None:
    assert beta_lower_bound(1, 1) == pytest.approx(0.2113, abs=1e-3)
    assert beta_lower_bound(10, 1) > beta_lower_bound(2, 1)
    a, b = decay(11.0, 1.0, days=35.0, lam=0.98)
    assert 1.0 < a < 11.0 and b == pytest.approx(1.0)
    # decay is toward the prior, never past it
    a2, _ = decay(11.0, 1.0, days=10_000, lam=0.98)
    assert a2 == pytest.approx(1.0, abs=1e-3)


def test_credit_on_success(learner: Learner, fake_llm: FakeLLM) -> None:
    lesson_id = _seed_lesson(learner, fake_llm, "When doing X", "Do Y first.")
    _run_with_lesson(learner, "doing X again", success=True)
    lesson = learner._store.get_lesson(lesson_id)
    assert lesson.alpha == pytest.approx(2.0)  # single injection -> full credit
    assert lesson.beta == pytest.approx(1.0)
    assert lesson.times_injected == 1


def test_credit_on_failure(learner: Learner, fake_llm: FakeLLM) -> None:
    lesson_id = _seed_lesson(learner, fake_llm, "When doing X", "Do Y first.")
    _run_with_lesson(learner, "doing X again", success=False)
    lesson = learner._store.get_lesson(lesson_id)
    assert lesson.alpha == pytest.approx(1.0)
    assert lesson.beta == pytest.approx(2.0)


def test_each_injected_lesson_gets_full_trial(learner: Learner, fake_llm: FakeLLM) -> None:
    a = _seed_lesson(learner, fake_llm, "When doing X", "Do Y.")
    b = _seed_lesson(learner, fake_llm, "When parsing dates in the report", "Use ISO format.")
    _run_with_lesson(learner, "doing X with dates in the report", success=True)
    la, lb = learner._store.get_lesson(a), learner._store.get_lesson(b)
    injected = [lesson for lesson in (la, lb) if lesson.times_injected]
    assert injected, "at least one lesson should have been injected"
    for lesson in injected:
        # every injection is one full Bernoulli trial for that lesson
        assert lesson.alpha == pytest.approx(2.0)
        assert lesson.beta == pytest.approx(1.0)


def test_feedback_reverses_credit(learner: Learner, fake_llm: FakeLLM) -> None:
    lesson_id = _seed_lesson(learner, fake_llm, "When doing X", "Do Y first.")
    with learner.run(task="doing X again") as run:
        _ = run.lessons
        run.end(success=True)
    assert learner._store.get_lesson(lesson_id).alpha == pytest.approx(2.0)
    # two hours later the user says it was actually wrong
    learner.feedback(run.id, "that answer was wrong", success=False)
    lesson = learner._store.get_lesson(lesson_id)
    assert lesson.alpha == pytest.approx(1.0)
    assert lesson.beta == pytest.approx(2.0)


def test_promotion(learner: Learner, fake_llm: FakeLLM) -> None:
    lesson_id = _seed_lesson(learner, fake_llm, "When doing X", "Do Y first.")
    for i in range(3):
        _run_with_lesson(learner, f"doing X round {i}", success=True)
    lesson = learner._store.get_lesson(lesson_id)
    assert lesson.state == "active"


def test_retirement_of_harmful_candidate(learner: Learner, fake_llm: FakeLLM) -> None:
    _seed_baseline(learner, n_success=20)  # healthy agent: baseline ~0.95
    lesson_id = _seed_lesson(learner, fake_llm, "When doing X", "Do the wrong thing.")
    for i in range(5):
        _run_with_lesson(learner, f"doing X round {i}", success=False)
    lesson = learner._store.get_lesson(lesson_id)
    assert lesson.state == "retired"


def test_cap_prunes_weakest(fake_llm: FakeLLM) -> None:
    learner = Learner(store=":memory:", embedder="hashed", llm=fake_llm, max_lessons=2)
    for condition, advice in [
        ("When submitting payment forms", "Send amounts in integer cents."),
        ("When parsing CSV exports", "Skip the header row."),
        ("When calling the search endpoint", "Paginate with a cursor, not offsets."),
    ]:
        _seed_lesson(learner, fake_llm, condition, advice)
    non_retired = [
        lesson for lesson in learner.lessons() if lesson.state in ("candidate", "active")
    ]
    assert len(non_retired) == 2


def test_forget(learner: Learner, fake_llm: FakeLLM) -> None:
    lesson_id = _seed_lesson(learner, fake_llm, "When doing X", "Do Y.")
    learner.forget(lesson_id)
    assert learner._store.get_lesson(lesson_id).state == "retired"
    assert learner.recall("doing X").lessons == []


def test_stats(learner: Learner, fake_llm: FakeLLM) -> None:
    _seed_lesson(learner, fake_llm, "When doing X", "Do Y.")
    stats = learner.stats()
    assert stats.lessons_by_state["candidate"] == 1
    assert 0.0 <= stats.baseline_success_rate <= 1.0
    assert stats.injected_tokens_estimate == 0  # seeded but never injected

    with learner.run(task="doing X again") as run:
        assert run.lessons  # injection happens here
        run.end(success=True)
    stats = learner.stats()
    assert stats.injected_tokens_estimate > 0
