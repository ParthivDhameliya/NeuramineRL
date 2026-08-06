"""Async facade: the sync API, thread-offloaded.

``AsyncLearner``/``AsyncRun`` mirror ``Learner``/``Run`` one-to-one but run
every blocking operation (SQLite/Postgres IO, and the reflection LLM call
inside ``end()``) in a worker thread via ``asyncio.to_thread``, so
asyncio-native agents (LangGraph, FastAPI handlers, ...) never block the
event loop. The underlying components are the sync ones; native-async
internals can replace the offload later without changing this API.

The one deliberate difference: ``run.lessons`` is a property on the sync
``Run`` but an awaitable method here (``await run.lessons()``), because the
first access performs retrieval IO.
"""

from __future__ import annotations

import asyncio
from types import TracebackType
from typing import Any, Literal

from .learner import Learner, LearnerStats
from .models import Lesson, LessonState
from .run import RecallResult, Run

__all__ = ["AsyncLearner", "AsyncRun"]


class AsyncRun:
    """Thread-offloading wrapper around a sync ``Run``. Use as an async
    context manager: an unhandled exception is recorded as a failure and
    re-raised, exactly like the sync version."""

    def __init__(self, run: Run) -> None:
        self._run = run

    @property
    def id(self) -> str:
        return self._run.id

    @property
    def task(self) -> str:
        return self._run.task

    async def lessons(self) -> RecallResult:
        return await asyncio.to_thread(lambda: self._run.lessons)

    async def log(self, messages: Any) -> None:
        await asyncio.to_thread(self._run.log, messages)

    async def log_tool_call(
        self, name: str, args: Any, result: Any = None, *, error: str | None = None
    ) -> None:
        await asyncio.to_thread(lambda: self._run.log_tool_call(name, args, result, error=error))

    async def note(self, text: str) -> None:
        await asyncio.to_thread(self._run.note, text)

    async def end(
        self,
        *,
        success: bool | None = None,
        score: float | None = None,
        error: str | None = None,
        detail: str = "",
    ) -> None:
        await asyncio.to_thread(
            lambda: self._run.end(success=success, score=score, error=error, detail=detail)
        )

    async def __aenter__(self) -> AsyncRun:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> Literal[False]:
        await asyncio.to_thread(self._run.__exit__, exc_type, exc, tb)
        return False  # never swallow exceptions


class AsyncLearner:
    """Thread-offloading wrapper around a sync ``Learner``.

    Construct it with the same keyword arguments as ``Learner`` (construction
    itself is synchronous and cheap), or wrap an existing instance::

        nm = AsyncLearner(scope="checkout-agent")
        async with await nm.run(task="...") as run:
            block = await run.lessons()
            ...
            await run.end(success=True)
    """

    def __init__(self, learner: Learner | None = None, **kwargs: Any) -> None:
        self._learner = learner if learner is not None else Learner(**kwargs)

    @property
    def sync(self) -> Learner:
        """The wrapped sync Learner, for anything not mirrored here."""
        return self._learner

    async def run(self, task: str, *, metadata: dict[str, Any] | None = None) -> AsyncRun:
        run = await asyncio.to_thread(lambda: self._learner.run(task, metadata=metadata))
        return AsyncRun(run)

    async def recall(self, task: str) -> RecallResult:
        return await asyncio.to_thread(self._learner.recall, task)

    async def feedback(self, run_id: str, note: str, *, success: bool) -> None:
        await asyncio.to_thread(lambda: self._learner.feedback(run_id, note, success=success))

    async def learn(self, run_id: str) -> list[Lesson]:
        return await asyncio.to_thread(self._learner.learn, run_id)

    async def lessons(self, state: LessonState | None = None) -> list[Lesson]:
        return await asyncio.to_thread(self._learner.lessons, state)

    async def forget(self, lesson_id: str) -> None:
        await asyncio.to_thread(self._learner.forget, lesson_id)

    async def stats(self) -> LearnerStats:
        return await asyncio.to_thread(self._learner.stats)

    async def close(self) -> None:
        await asyncio.to_thread(self._learner.close)
