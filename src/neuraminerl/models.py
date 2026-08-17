"""Core data model.

The unit of value is the Lesson; trajectories, outcomes, and injections exist
as evidence for lessons. Every lesson is traceable to the incidents that
created it (``source_trajectory_ids``) and the runs it influenced (injections).
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal

TrajectoryStatus = Literal["running", "completed", "abandoned"]
OutcomeStatus = Literal["success", "failure", "partial", "unknown"]
OutcomeSource = Literal["exception", "manual", "user_correction", "eval", "llm_judge"]
LessonState = Literal["candidate", "active", "retired", "merged"]
StepKind = Literal["llm_call", "tool_call", "tool_result", "user_message", "agent_message", "note"]


def new_id() -> str:
    """Time-sortable unique id: millisecond timestamp prefix + random suffix."""
    return f"{int(time.time() * 1000):013x}{uuid.uuid4().hex[:16]}"


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


# Hard caps on lesson text, applied wherever a lesson is written — including
# the merge path, where the model rewrites both fields. An oversized lesson is
# silently un-injectable (it never fits the injector's budget) yet accrues no
# evidence, so it can never be retired either.
MAX_ADVICE_CHARS = 240
MAX_CONDITION_CHARS = 160


@dataclass
class Trajectory:
    scope: str
    task: str
    id: str = field(default_factory=new_id)
    status: TrajectoryStatus = "running"
    started_at: str = field(default_factory=utcnow)
    ended_at: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    steps: list[Step] = field(default_factory=list)


@dataclass
class Step:
    trajectory_id: str
    index: int
    kind: StepKind
    content: str
    error: str | None = None
    id: str = field(default_factory=new_id)
    created_at: str = field(default_factory=utcnow)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Outcome:
    trajectory_id: str
    status: OutcomeStatus
    source: OutcomeSource
    score: float | None = None
    detail: str = ""
    id: str = field(default_factory=new_id)
    created_at: str = field(default_factory=utcnow)


@dataclass
class Lesson:
    scope: str
    condition: str
    advice: str
    rationale: str = ""
    state: LessonState = "candidate"
    alpha: float = 1.0
    beta: float = 1.0
    times_injected: int = 0
    credited_trials: float = 0.0
    """Outcomes actually credited to this lesson, monotone and never decayed.

    Lifecycle thresholds count this, not ``times_injected`` (which counts
    exposure and is bumped at retrieval, before any outcome exists) and not
    ``alpha+beta-2`` (which decay shrinks, so with injections more than about a
    week apart the count saturates below the thresholds and a failing lesson
    could never be retired).
    """
    version: int = 1
    merged_into: str | None = None
    source_trajectory_ids: list[str] = field(default_factory=list)
    last_injected_at: str | None = None
    last_reinforced_at: str | None = None
    last_decay_at: str = field(default_factory=utcnow)
    id: str = field(default_factory=new_id)
    created_at: str = field(default_factory=utcnow)
    updated_at: str = field(default_factory=utcnow)

    @property
    def text(self) -> str:
        """The canonical text used for embedding and injection."""
        return f"{self.condition.rstrip('.')}: {self.advice}"


@dataclass
class Injection:
    """The credit-assignment edge: lesson L was injected into trajectory T.

    ``credited_alpha``/``credited_beta`` record exactly what this injection
    added to the lesson's evidence, so a delayed correction can reverse and
    re-apply credit idempotently.
    """

    trajectory_id: str
    lesson_id: str
    lesson_version: int
    retrieval_score: float
    rank: int
    credited: bool = False
    credited_alpha: float = 0.0
    credited_beta: float = 0.0
    id: str = field(default_factory=new_id)
    created_at: str = field(default_factory=utcnow)


@dataclass
class LessonDraft:
    """A candidate lesson produced by reflection, before dedup/merge."""

    condition: str
    advice: str
    rationale: str = ""
