from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol, runtime_checkable

from ..embeddings.base import Vector
from ..exceptions import ConfigError
from ..models import Injection, Lesson, LessonState, Outcome, Step, Trajectory


def check_query_dim(matrix: Any, query_vec: Vector, scope: str) -> None:
    """Raise an actionable error before numpy does an unreadable one.

    The write-time guard cannot catch this on its own: retrieval happens before
    the first write of a run, so a mismatched embedder surfaces here first, as
    ``matmul: Input operand 1 has a mismatch in its core dimension``.
    """
    if getattr(matrix, "size", 0) and matrix.shape[1] != query_vec.shape[-1]:
        raise ConfigError(
            f"Scope '{scope}' stores {matrix.shape[1]}-dim lesson embeddings but the current "
            f"embedder produces {query_vec.shape[-1]}-dim vectors. Use the embedder this scope "
            f"was built with, or start a new scope to re-learn."
        )


@runtime_checkable
class Store(Protocol):
    """Persistence boundary. The SQLite default hides brute-force vector
    search behind ``search_lessons`` so pgvector/sqlite-vec can slot in later
    without API changes."""

    def save_trajectory(self, trajectory: Trajectory) -> None: ...

    def get_trajectory(self, trajectory_id: str, *, with_steps: bool = True) -> Trajectory: ...

    def add_steps(self, steps: Sequence[Step]) -> None: ...

    def save_outcome(self, outcome: Outcome) -> None: ...

    def latest_outcome(self, trajectory_id: str) -> Outcome | None: ...

    def upsert_lesson(self, lesson: Lesson, embedding: Vector | None = None) -> None: ...

    def get_lesson(self, lesson_id: str) -> Lesson: ...

    def get_lessons(self, scope: str, states: Sequence[LessonState]) -> list[Lesson]: ...

    def count_lessons(self, scope: str, states: Sequence[LessonState]) -> int: ...

    def search_lessons(
        self, query_vec: Vector, scope: str, states: Sequence[LessonState], k: int
    ) -> list[tuple[Lesson, float]]: ...

    def log_injections(self, injections: Sequence[Injection]) -> None: ...

    def injections_for(self, trajectory_id: str) -> list[Injection]: ...

    def update_injection(self, injection: Injection) -> None: ...

    def baseline_success_rate(self, scope: str, window: int = 100) -> float: ...

    def log_event(self, lesson_id: str, event: str, detail: str = "") -> None: ...

    def close(self) -> None: ...
