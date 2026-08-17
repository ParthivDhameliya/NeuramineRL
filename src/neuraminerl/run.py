from __future__ import annotations

import json
import threading
import traceback
from types import TracebackType
from typing import TYPE_CHECKING, Any, Literal

from .models import Lesson, Step, StepKind, Trajectory, utcnow
from .retrieval.injector import estimate_tokens

if TYPE_CHECKING:
    from .learner import Learner

_ROLE_TO_KIND: dict[str, StepKind] = {
    "user": "user_message",
    "assistant": "agent_message",
    "system": "note",
    "tool": "tool_result",
    "function": "tool_result",
}


def _content_to_text(content: Any) -> str:
    """Flatten OpenAI/Anthropic-style message content into plain text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):  # content blocks
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict):
                if isinstance(block.get("text"), str):
                    parts.append(block["text"])
                else:
                    parts.append(json.dumps(block, default=str))
            else:
                parts.append(str(block))
        return "\n".join(parts)
    return json.dumps(content, default=str)


class RecallResult:
    """Lessons recalled for a task. ``str()`` renders the injectable prompt
    block (empty string when there are no lessons)."""

    def __init__(self, lessons: list[Lesson], block: str) -> None:
        self.lessons = lessons
        self.block = block

    def __str__(self) -> str:
        return self.block

    def __bool__(self) -> bool:
        return bool(self.lessons)

    def __len__(self) -> int:
        return len(self.lessons)

    @property
    def token_count(self) -> int:
        return estimate_tokens(self.block) if self.block else 0


class Run:
    """One agent attempt at one task. Use as a context manager: an unhandled
    exception is recorded as a failure and re-raised."""

    def __init__(self, learner: Learner, trajectory: Trajectory) -> None:
        self._learner = learner
        self._trajectory = trajectory
        self._step_index = 0
        self._lessons: RecallResult | None = None
        self._ended = False
        # AsyncRun runs this object's methods in worker threads, so the
        # recall memoization below needs real mutual exclusion: two concurrent
        # first-accesses would each bind the run, logging duplicate injections
        # and crediting one run's outcome twice.
        self._lock = threading.Lock()

    @property
    def id(self) -> str:
        return self._trajectory.id

    @property
    def task(self) -> str:
        return self._trajectory.task

    @property
    def lessons(self) -> RecallResult:
        """Recalled lessons for this task. First access performs retrieval and
        binds the injected lessons to this run for credit assignment."""
        if self._lessons is None:
            with self._lock:
                if self._lessons is None:
                    self._lessons = self._learner._recall_for_run(self._trajectory)
        return self._lessons

    # -- capture -----------------------------------------------------------

    def log(self, messages: Any) -> None:
        """Best-effort trajectory capture. Accepts a string, one message dict,
        or a list of OpenAI/Anthropic-shaped message dicts."""
        if isinstance(messages, (str, dict)):
            messages = [messages]
        steps: list[Step] = []
        for message in messages:
            if isinstance(message, str):
                kind: StepKind = "note"
                content = message
            else:
                role = str(message.get("role", "note"))
                kind = _ROLE_TO_KIND.get(role, "note")
                content = _content_to_text(message.get("content"))
            steps.append(self._make_step(kind, content))
        self._learner._store.add_steps(steps)
        self._trajectory.steps.extend(steps)

    def log_tool_call(
        self, name: str, args: Any, result: Any = None, *, error: str | None = None
    ) -> None:
        call = self._make_step("tool_call", f"{name}({json.dumps(args, default=str)})")
        result_step = self._make_step(
            "tool_result", _content_to_text(result) if result is not None else "", error=error
        )
        self._learner._store.add_steps([call, result_step])
        self._trajectory.steps.extend([call, result_step])

    def note(self, text: str) -> None:
        step = self._make_step("note", text)
        self._learner._store.add_steps([step])
        self._trajectory.steps.append(step)

    def _make_step(self, kind: StepKind, content: str, error: str | None = None) -> Step:
        cap = self._learner.config.max_step_chars
        step = Step(
            trajectory_id=self._trajectory.id,
            index=self._step_index,
            kind=kind,
            content=content[:cap],
            error=error[:cap] if error else None,
        )
        self._step_index += 1
        return step

    # -- outcome -----------------------------------------------------------

    def end(
        self,
        *,
        success: bool | None = None,
        score: float | None = None,
        error: str | None = None,
        detail: str = "",
    ) -> None:
        """Record the outcome. On failure (and reflect="sync") this triggers
        one reflection LLM call — after the run, never on the hot path.

        Passing ``error`` without ``success`` means failure: reporting an error
        and having it recorded as an outcome that teaches nothing would be a
        silent no-op. ``detail`` alone stays neutral.

        ``score`` is a probability of success in [0.0, 1.0]. It becomes Beta
        evidence directly, so a value on some other scale (0-10, 0-100) would
        write evidence weaker than the prior; that is rejected here rather than
        silently corrupting the lesson's health.
        """
        if score is not None and not 0.0 <= score <= 1.0:
            raise ValueError(f"score must be between 0.0 and 1.0, got {score!r}")
        with self._lock:
            if self._ended:
                return
            self._ended = True
        if success is None and error:
            success = False
        self._learner._finish(
            self._trajectory,
            success=success,
            score=score,
            detail=error or detail,
            source="manual",
        )

    def __enter__(self) -> Run:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> Literal[False]:
        with self._lock:
            already_ended = self._ended
            self._ended = True
        if exc is not None and not already_ended:
            self._learner._finish(
                self._trajectory,
                success=False,
                score=None,
                detail="".join(traceback.format_exception(exc_type, exc, tb))[-4000:],
                source="exception",
            )
        elif not already_ended:
            # Exited without end(): nothing to learn from, but don't lose the trace.
            self._trajectory.status = "abandoned"
            self._trajectory.ended_at = utcnow()
            self._learner._store.save_trajectory(self._trajectory)
        return False  # never swallow exceptions
