from __future__ import annotations

import pytest

from neuramine.embeddings.hashed import HashedEmbedder
from neuramine.exceptions import ConfigError, NeuramineError
from neuramine.models import Injection, Lesson, Outcome, Step, Trajectory
from neuramine.store.sqlite import SqliteStore


@pytest.fixture
def store() -> SqliteStore:
    return SqliteStore(":memory:")


@pytest.fixture
def emb() -> HashedEmbedder:
    return HashedEmbedder()


def _lesson(emb: HashedEmbedder, condition: str, advice: str, **kw: object) -> Lesson:
    return Lesson(scope="default", condition=condition, advice=advice, **kw)  # type: ignore[arg-type]


def test_trajectory_roundtrip(store: SqliteStore) -> None:
    t = Trajectory(scope="default", task="do the thing", metadata={"model": "x"})
    store.save_trajectory(t)
    store.add_steps(
        [
            Step(trajectory_id=t.id, index=0, kind="user_message", content="hello"),
            Step(
                trajectory_id=t.id, index=1, kind="tool_call", content="search(q=1)", error="boom"
            ),
        ]
    )
    loaded = store.get_trajectory(t.id)
    assert loaded.task == "do the thing"
    assert loaded.metadata == {"model": "x"}
    assert [s.kind for s in loaded.steps] == ["user_message", "tool_call"]
    assert loaded.steps[1].error == "boom"


def test_unknown_trajectory_raises(store: SqliteStore) -> None:
    with pytest.raises(NeuramineError):
        store.get_trajectory("nope")


def test_lesson_roundtrip_and_search(store: SqliteStore, emb: HashedEmbedder) -> None:
    l1 = _lesson(emb, "When submitting the booking form", "Use ISO dates")
    l2 = _lesson(emb, "When calling the payments API", "Send amounts in integer cents")
    store.upsert_lesson(l1, embedding=emb.embed([l1.text])[0])
    store.upsert_lesson(l2, embedding=emb.embed([l2.text])[0])

    query = emb.embed(["submit the booking form with dates"])[0]
    results = store.search_lessons(query, "default", states=("candidate",), k=2)
    assert results[0][0].id == l1.id
    assert results[0][1] > results[1][1]


def test_new_lesson_requires_embedding(store: SqliteStore) -> None:
    with pytest.raises(ConfigError):
        store.upsert_lesson(Lesson(scope="default", condition="c", advice="a"))


def test_embedding_dim_mismatch_rejected(store: SqliteStore, emb: HashedEmbedder) -> None:
    l1 = _lesson(emb, "cond", "advice")
    store.upsert_lesson(l1, embedding=emb.embed([l1.text])[0])
    other = HashedEmbedder(dim=64)
    l2 = _lesson(emb, "cond2", "advice2")
    with pytest.raises(ConfigError):
        store.upsert_lesson(l2, embedding=other.embed([l2.text])[0])


def test_update_preserves_embedding(store: SqliteStore, emb: HashedEmbedder) -> None:
    lesson = _lesson(emb, "cond", "advice")
    store.upsert_lesson(lesson, embedding=emb.embed([lesson.text])[0])
    lesson.alpha = 5.0
    store.upsert_lesson(lesson)  # no embedding passed
    query = emb.embed([lesson.text])[0]
    results = store.search_lessons(query, "default", states=("candidate",), k=1)
    assert results and results[0][0].alpha == 5.0


def test_injection_roundtrip(store: SqliteStore, emb: HashedEmbedder) -> None:
    t = Trajectory(scope="default", task="x")
    store.save_trajectory(t)
    lesson = _lesson(emb, "c", "a")
    store.upsert_lesson(lesson, embedding=emb.embed([lesson.text])[0])
    inj = Injection(
        trajectory_id=t.id, lesson_id=lesson.id, lesson_version=1, retrieval_score=0.9, rank=0
    )
    store.log_injections([inj])
    loaded = store.injections_for(t.id)
    assert len(loaded) == 1 and not loaded[0].credited
    inj.credited = True
    inj.credited_alpha = 0.5
    store.update_injection(inj)
    assert store.injections_for(t.id)[0].credited_alpha == 0.5


def test_baseline(store: SqliteStore) -> None:
    # fewer than 5 outcomes -> 0.5 default
    assert store.baseline_success_rate("default") == 0.5
    for i in range(10):
        t = Trajectory(scope="default", task=f"t{i}")
        store.save_trajectory(t)
        store.save_outcome(
            Outcome(
                trajectory_id=t.id,
                status="success" if i < 8 else "failure",
                source="manual",
            )
        )
    assert store.baseline_success_rate("default") == pytest.approx(0.8)


def test_baseline_uses_latest_outcome_per_trajectory(store: SqliteStore) -> None:
    for i in range(6):
        t = Trajectory(scope="default", task=f"t{i}")
        store.save_trajectory(t)
        store.save_outcome(Outcome(trajectory_id=t.id, status="failure", source="manual"))
        # delayed correction flips it to success
        store.save_outcome(Outcome(trajectory_id=t.id, status="success", source="user_correction"))
    assert store.baseline_success_rate("default") == pytest.approx(1.0)
