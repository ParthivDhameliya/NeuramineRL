from __future__ import annotations

import warnings
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from .config import LearnerConfig
from .embeddings.base import Embedder
from .embeddings.hashed import HashedEmbedder
from .embeddings.local import LocalEmbedder
from .lessons.lifecycle import Lifecycle
from .llm import auto as llm_auto
from .llm.base import LLMClient
from .models import (
    Injection,
    Lesson,
    LessonState,
    Outcome,
    OutcomeSource,
    OutcomeStatus,
    Step,
    Trajectory,
    utcnow,
)
from .reflection.dedup import Deduplicator
from .reflection.llm_reflector import FallbackReflector, LLMReflector
from .retrieval.injector import Injector, estimate_tokens
from .retrieval.retriever import Retriever
from .run import RecallResult, Run
from .store.base import Store
from .store.sqlite import SqliteStore


@dataclass
class LearnerStats:
    scope: str
    baseline_success_rate: float
    lessons_by_state: dict[str, int]
    injected_tokens_estimate: int
    """Rough cumulative input tokens this scope's lesson injections have cost
    across all runs (current lesson text x times injected, ~4 chars/token)."""


class Learner:
    """The facade. Zero-config: SQLite + local embeddings in ./.neuraminerl/,
    reflection LLM auto-detected from ANTHROPIC_API_KEY / OPENAI_API_KEY."""

    def __init__(
        self,
        *,
        store: Store | str | Path | None = None,
        embedder: Embedder | str | None = None,
        llm: LLMClient | str | None = None,
        scope: str = "default",
        config: LearnerConfig | None = None,
        **overrides: Any,
    ) -> None:
        base = config or LearnerConfig()
        self.config = replace(base, **overrides) if overrides else base
        self.scope = scope

        if store is None:
            self._store: Store = SqliteStore(self.config.home / "neuraminerl.db")
        elif isinstance(store, (str, Path)):
            self._store = SqliteStore(store)
        else:
            self._store = store

        if embedder is None:
            self._embedder = self._default_embedder()
        elif isinstance(embedder, str):
            self._embedder = HashedEmbedder() if embedder == "hashed" else LocalEmbedder(embedder)
        else:
            self._embedder = embedder

        if llm is None:
            self._llm: LLMClient | None = llm_auto.detect()
            if self._llm is None:
                warnings.warn(
                    "No reflection LLM configured (set ANTHROPIC_API_KEY or OPENAI_API_KEY, "
                    "or pass llm=...). Failures will be stored as raw observations instead "
                    "of distilled lessons.",
                    stacklevel=2,
                )
        elif isinstance(llm, str):
            self._llm = llm_auto.from_spec(llm)
        else:
            self._llm = llm

        self._reflector = LLMReflector(self._llm, self.config) if self._llm else FallbackReflector()
        self._dedup = Deduplicator(self._store, self._embedder, self._llm, self.config)
        self._retriever = Retriever(self._store, self._embedder, self.config)
        self._injector = Injector(self.config)
        self._lifecycle = Lifecycle(self._store, self.config)

    @staticmethod
    def _default_embedder() -> Embedder:
        try:
            import model2vec  # noqa: F401

            return LocalEmbedder()
        except ImportError:
            warnings.warn(
                "model2vec is not installed; falling back to a crude hashing embedder. "
                "For better lesson retrieval: pip install neuraminerl[embeddings]",
                stacklevel=3,
            )
            return HashedEmbedder()

    # -- the loop ------------------------------------------------------------

    def run(self, task: str, *, metadata: dict[str, Any] | None = None) -> Run:
        trajectory = Trajectory(scope=self.scope, task=task, metadata=metadata or {})
        self._store.save_trajectory(trajectory)
        return Run(self, trajectory)

    def recall(self, task: str) -> RecallResult:
        """Read-only recall: renders the lesson block WITHOUT binding lessons
        to a run. Use ``run.lessons`` inside a run so outcomes feed back."""
        ranked = self._retriever.retrieve(task, self.scope)
        block, included = self._injector.render([lesson for lesson, _ in ranked])
        return RecallResult(included, block)

    def feedback(self, run_id: str, note: str, *, success: bool) -> None:
        """Delayed outcome ("the user said this was wrong two hours later").
        Reverses and re-applies credit for the run's injected lessons, and
        re-reflects with the note as evidence when it's a failure."""
        trajectory = self._store.get_trajectory(run_id)
        step = Step(
            trajectory_id=run_id,
            index=len(trajectory.steps),
            kind="user_message",
            content=f"[delayed feedback] {note}",
        )
        self._store.add_steps([step])
        trajectory.steps.append(step)
        outcome = Outcome(
            trajectory_id=run_id,
            status="success" if success else "failure",
            source="user_correction",
            detail=note,
        )
        self._store.save_outcome(outcome)
        self._lifecycle.credit(outcome)
        if not success and self.config.reflect == "sync":
            self._reflect(trajectory, outcome)
        self._lifecycle.maintain(self.scope)

    def learn(self, run_id: str) -> list[Lesson]:
        """Force reflection on a past run (e.g. after reflect='off')."""
        trajectory = self._store.get_trajectory(run_id)
        outcome = self._store.latest_outcome(run_id)
        if outcome is None:
            outcome = Outcome(trajectory_id=run_id, status="unknown", source="manual")
        return self._reflect(trajectory, outcome)

    # -- inspection / control -------------------------------------------------

    def lessons(self, state: LessonState | None = None) -> list[Lesson]:
        states: tuple[LessonState, ...] = (state,) if state else ("candidate", "active", "retired")
        return self._store.get_lessons(self.scope, states)

    def forget(self, lesson_id: str) -> None:
        lesson = self._store.get_lesson(lesson_id)
        lesson.state = "retired"
        self._store.upsert_lesson(lesson)
        self._store.log_event(lesson_id, "forgotten", "retired by user")

    def stats(self) -> LearnerStats:
        by_state = {
            state: self._store.count_lessons(self.scope, (state,))
            for state in ("candidate", "active", "retired")
        }
        lessons = self._store.get_lessons(self.scope, ("candidate", "active", "retired"))
        injected_tokens = sum(
            estimate_tokens(lesson.text) * lesson.times_injected
            for lesson in lessons
            if lesson.times_injected
        )
        return LearnerStats(
            scope=self.scope,
            baseline_success_rate=self._store.baseline_success_rate(
                self.scope, self.config.baseline_window
            ),
            lessons_by_state=by_state,
            injected_tokens_estimate=injected_tokens,
        )

    def close(self) -> None:
        self._store.close()

    # -- internals (called by Run) ---------------------------------------------

    def _recall_for_run(self, trajectory: Trajectory) -> RecallResult:
        ranked = self._retriever.retrieve(trajectory.task, self.scope)
        scores = {lesson.id: score for lesson, score in ranked}
        block, included = self._injector.render([lesson for lesson, _ in ranked])
        injections = []
        now = utcnow()
        for rank, lesson in enumerate(included):
            injections.append(
                Injection(
                    trajectory_id=trajectory.id,
                    lesson_id=lesson.id,
                    lesson_version=lesson.version,
                    retrieval_score=scores[lesson.id],
                    rank=rank,
                )
            )
            lesson.times_injected += 1
            lesson.last_injected_at = now
            self._store.upsert_lesson(lesson)
        self._store.log_injections(injections)
        return RecallResult(included, block)

    def _finish(
        self,
        trajectory: Trajectory,
        *,
        success: bool | None,
        score: float | None,
        detail: str,
        source: OutcomeSource,
    ) -> None:
        status: OutcomeStatus
        if success is True:
            status = "success"
        elif success is False:
            status = "failure"
        elif score is not None:
            status = "partial"
        else:
            status = "unknown"
        trajectory.status = "completed"
        trajectory.ended_at = utcnow()
        self._store.save_trajectory(trajectory)
        outcome = Outcome(
            trajectory_id=trajectory.id, status=status, score=score, source=source, detail=detail
        )
        self._store.save_outcome(outcome)
        self._lifecycle.credit(outcome)
        if status == "failure" and self.config.reflect == "sync":
            self._reflect(trajectory, outcome)
        self._lifecycle.maintain(self.scope)

    def _reflect(self, trajectory: Trajectory, outcome: Outcome) -> list[Lesson]:
        if not trajectory.steps:
            trajectory = self._store.get_trajectory(trajectory.id)
        drafts = self._reflector.reflect(trajectory, outcome)
        return [self._dedup.resolve(draft, self.scope, trajectory.id) for draft in drafts]
