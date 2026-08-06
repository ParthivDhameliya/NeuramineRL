from __future__ import annotations

import asyncio

import pytest

from neuraminerl import AsyncLearner
from neuraminerl.llm.fake import FakeLLM

REFLECTION = {
    "lessons": [
        {"condition": "When testing", "advice": "Do the thing right.", "rationale": "it broke"}
    ]
}


def test_async_learner_full_loop() -> None:
    async def main() -> None:
        fake = FakeLLM(responses=[REFLECTION])
        nm = AsyncLearner(store=":memory:", embedder="hashed", llm=fake)

        # Episode 1: failure -> reflection -> lesson stored.
        async with await nm.run(task="test the thing") as run:
            await run.log("attempt one")
            await run.note("something odd")
            await run.end(success=False, error="boom")
        assert await nm.lessons()

        # Episode 2: the lesson is recalled and injected.
        async with await nm.run(task="test the thing") as run:
            recall = await run.lessons()
            assert recall
            assert "Do the thing right." in str(recall)
            await run.end(success=True)

        stats = await nm.stats()
        assert stats.injected_tokens_estimate > 0
        assert (await nm.recall("test the thing")).lessons
        await nm.close()

    asyncio.run(main())


def test_async_run_records_unhandled_exception() -> None:
    async def main() -> None:
        nm = AsyncLearner(store=":memory:", embedder="hashed", llm=FakeLLM())
        run_id = ""
        with pytest.raises(ValueError, match="boom"):
            async with await nm.run(task="explode") as run:
                run_id = run.id
                raise ValueError("boom")
        outcome = nm.sync._store.latest_outcome(run_id)
        assert outcome is not None
        assert outcome.status == "failure"
        await nm.close()

    asyncio.run(main())


def test_async_wraps_existing_learner() -> None:
    from neuraminerl import Learner

    sync_learner = Learner(store=":memory:", embedder="hashed", llm=FakeLLM())
    nm = AsyncLearner(sync_learner)
    assert nm.sync is sync_learner

    async def main() -> None:
        async with await nm.run(task="t") as run:
            await run.end(success=True)
        stats = await nm.stats()
        assert stats.scope == "default"

    asyncio.run(main())
